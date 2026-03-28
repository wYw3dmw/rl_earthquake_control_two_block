from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This module requires gymnasium. Install it with `pip install gymnasium`."
    ) from exc

try:
    from scipy.integrate import solve_ivp
except Exception:
    solve_ivp = None


@dataclass
class TwoBlockPressureParams:
    # Core nondimensional parameters from the existing simulation
    A: float = 10.0
    B: float = 1.0
    C: float = 83.2
    mu_s: float = 0.6
    mu_k: float = 0.5
    eps: float = 1e-6

    # Control-related parameters
    pressure_scale: float = 1.0
    P_max: float = 0.95
    dP: float = 0.01

    # Integration / RL timing
    dt: float = 1e-3
    n_substeps: int = 10
    t_max: float = 5.0
    v_fail: float = 20.0

    # Old reward params (kept for compatibility; not used by the new reward)
    w_U: float = 1.0
    w_K: float = 2.0
    U_ref: float = 400.0
    K_ref: float = 300.0
    reward_power_U: float = 1.0
    reward_power_K: float = 1.5

    # New reward settings
    reward_eps: float = 1e-8
    reward_clip: float = 50.0

    # Gaussian-style reward parameters
    reward_av: float = 2.0
    reward_ax: float = 2.0
    reward_bv: float = 0.1
    reward_bx: float = 0.05

    # Observation normalization references
    x_ref: float = 1.0
    v_ref: float = 1.0
    t_ref: float = 1.0

    # Default reset state
    x1_0: float = 0.0
    x2_0: float = 0.1
    v1_0: float = 0.0
    v2_0: float = 0.0
    P1_0: float = 0.0
    P2_0: float = 0.0
    t0: float = 0.0

    # Optional hard step cap
    max_episode_steps: Optional[int] = None


class TwoBlockPressureEnv(gym.Env):
    """Gymnasium environment built directly from the user's existing ODE.

    Action space
    ------------
    MultiDiscrete([3, 3]) with per-block mapping:
        0 -> -dP
        1 ->  0
        2 -> +dP

    Observation
    -----------
    9D vector:
        [x1/x_ref, x2/x_ref,
         v1/v_ref, v2/v_ref,
         P1/pressure_scale, P2/pressure_scale,
         (x1-x2)/x_ref, (v1-v2)/v_ref,
         t/t_ref]

    Reward
    ------
        r = exp(-reward_av * (1 - reward_bv * v)**2 - reward_ax * (1 - reward_bx * x)**2)

    Termination
    -----------
    - terminate if max(|v1|, |v2|) > v_fail
    - truncate if t >= t_max
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, params: Optional[TwoBlockPressureParams] = None, seed: Optional[int] = None):
        super().__init__()
        self.params = params or TwoBlockPressureParams()
        self.action_space = spaces.MultiDiscrete([3, 3])

        obs_bound = np.full(9, np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_bound, obs_bound, dtype=np.float32)

        self.np_random = None
        self.state = np.zeros(4, dtype=np.float64)   # [x1, x2, v1, v2]
        self.pressure = np.zeros(2, dtype=np.float64)  # [P1, P2]
        self.t = 0.0
        self.step_count = 0
        self.last_action = np.array([1, 1], dtype=np.int64)

        self.reset(seed=seed)

    # ---------- physics ----------
    def friction_coefficients(self, x1: float, x2: float) -> Tuple[float, float]:
        p = self.params
        s1 = max(x1, 0.0)
        s2 = max(x2, 0.0)
        mu1 = p.mu_k + (p.mu_s - p.mu_k) * np.exp(-s1)
        mu2 = p.mu_k + (p.mu_s - p.mu_k) * np.exp(-s2)
        return mu1, mu2

    def friction_forces(
        self,
        x1: float,
        x2: float,
        v1: float,
        v2: float,
        P1: float,
        P2: float,
    ) -> Tuple[float, float]:
        p = self.params
        mu1, mu2 = self.friction_coefficients(x1, x2)
        sign1 = np.tanh(v1 / p.eps)
        sign2 = np.tanh(v2 / p.eps)

        reduction1 = np.clip(1.0 - P1 / p.pressure_scale, 0.0, None)
        reduction2 = np.clip(1.0 - P2 / p.pressure_scale, 0.0, None)

        Ff1 = p.C * mu1 * reduction1 * sign1
        Ff2 = p.C * mu2 * reduction2 * sign2
        return Ff1, Ff2

    def rhs(self, t: float, y: np.ndarray, pressure: Optional[np.ndarray] = None) -> np.ndarray:
        p = self.params
        x1, x2, v1, v2 = y
        P1, P2 = self.pressure if pressure is None else pressure

        Ff1, Ff2 = self.friction_forces(x1, x2, v1, v2, P1, P2)

        dx1dt = v1
        dx2dt = v2
        dv1dt = p.A * (10.0 * t + 10.0 - x1) + p.B * (x2 - x1) - Ff1
        dv2dt = p.A * (10.0 * t + 10.0 - x2) + p.B * (x1 - x2) - Ff2
        return np.array([dx1dt, dx2dt, dv1dt, dv2dt], dtype=np.float64)

    def _rk4_step(self, y: np.ndarray, pressure: np.ndarray, t: float, dt: float) -> np.ndarray:
        k1 = self.rhs(t, y, pressure)
        k2 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k1, pressure)
        k3 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k2, pressure)
        k4 = self.rhs(t + dt, y + dt * k3, pressure)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def loading_displacement(self, t: Optional[float] = None) -> float:
        tt = self.t if t is None else t
        return 10.0 * tt + 10.0

    def kinetic_energy(self, state: Optional[np.ndarray] = None) -> float:
        s = self.state if state is None else state
        _, _, v1, v2 = s
        return 0.5 * (v1 ** 2 + v2 ** 2)

    def potential_energy(self, state: Optional[np.ndarray] = None, t: Optional[float] = None) -> float:
        p = self.params
        s = self.state if state is None else state
        x1, x2, _, _ = s
        X = self.loading_displacement(t)
        return 0.5 * p.A * (X - x1) ** 2 + 0.5 * p.A * (X - x2) ** 2 + 0.5 * p.B * (x1 - x2) ** 2

    def total_energy(self, state: Optional[np.ndarray] = None, t: Optional[float] = None) -> float:
        return self.kinetic_energy(state) + self.potential_energy(state, t)

    def force_friction_ratios(
        self,
        state: Optional[np.ndarray] = None,
        pressure: Optional[np.ndarray] = None,
        t: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        ratio_i = |sum of driving/elastic forces on block i| / (|friction force_i| + eps)
        """
        p = self.params
        s = self.state if state is None else state
        pr = self.pressure if pressure is None else pressure
        tt = self.t if t is None else t

        x1, x2, v1, v2 = s
        P1, P2 = pr

        Ff1, Ff2 = self.friction_forces(x1, x2, v1, v2, P1, P2)

        Fe1 = p.A * (10.0 * tt + 10.0 - x1) + p.B * (x2 - x1)
        Fe2 = p.A * (10.0 * tt + 10.0 - x2) + p.B * (x1 - x2)

        ratio1 = abs(Ff1) / (abs(Fe1) + p.reward_eps)
        ratio2 = abs(Ff2) / (abs(Fe2) + p.reward_eps)
        return ratio1, ratio2

    # ---------- RL plumbing ----------
    def _action_to_delta_p(self, action: np.ndarray) -> np.ndarray:
        mapping = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        return mapping[action] * self.params.dP

    def _get_obs(self) -> np.ndarray:
        p = self.params
        x1, x2, v1, v2 = self.state
        P1, P2 = self.pressure
        K = self.kinetic_energy()
        U = self.potential_energy()

        return np.array([
            x1 / p.x_ref,
            x2 / p.x_ref,
            v1 / p.v_ref,
            v2 / p.v_ref,
            P1 / p.pressure_scale,
            P2 / p.pressure_scale,
            (x1 - x2) / p.x_ref,
            (v1 - v2) / p.v_ref,
            self.t / p.t_ref,
            K / p.K_ref,
            U / p.U_ref,
        ], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        ratio1, ratio2 = self.force_friction_ratios()
        mean_ratio = (ratio1 + ratio2) / 2.0

        return {
            "t": float(self.t),
            "x1": float(self.state[0]),
            "x2": float(self.state[1]),
            "v1": float(self.state[2]),
            "v2": float(self.state[3]),
            "P1": float(self.pressure[0]),
            "P2": float(self.pressure[1]),
            "kinetic_energy": float(self.kinetic_energy()),
            "potential_energy": float(self.potential_energy()),
            "total_energy": float(self.total_energy()),
            "force_friction_ratio_1": float(ratio1),
            "force_friction_ratio_2": float(ratio2),
            "mean_force_friction_ratio": float(mean_ratio),
        }

#    def compute_reward(self) -> float:
#        p = self.params
#        ratio1, ratio2 = self.force_friction_ratios()
#        mean_ratio = (ratio1 + ratio2) / 2.0

#        if hasattr(self, "prev_mean_ratio"):
#            reward = mean_ratio - self.prev_mean_ratio
#        else:
#            reward = 0.0

#        self.prev_mean_ratio = mean_ratio
#        return float(reward)

    def compute_reward(self) -> float:
        p = self.params
        x1, x2, v1, v2 = self.state

        v = max(abs(v1), abs(v2))
        x = max(abs(x1), abs(x2))

        reward = np.exp(
            -p.reward_av * (-1.0 + p.reward_bv * v) ** 2
            -p.reward_ax * (-1.0 + p.reward_bx * x) ** 2
        )
        return float(reward)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        p = self.params
        options = options or {}
        self.state = np.array([
            options.get("x1_0", p.x1_0),
            options.get("x2_0", p.x2_0),
            options.get("v1_0", p.v1_0),
            options.get("v2_0", p.v2_0),
        ], dtype=np.float64)
        self.pressure = np.array([
            options.get("P1_0", p.P1_0),
            options.get("P2_0", p.P2_0),
        ], dtype=np.float64)
        self.pressure = np.clip(self.pressure, 0.0, p.P_max)
        self.t = float(options.get("t0", p.t0))
        self.step_count = 0
        self.last_action = np.array([1, 1], dtype=np.int64)

        ratio1, ratio2 = self.force_friction_ratios()
        self.prev_mean_ratio = (ratio1 + ratio2) / 2.0

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        p = self.params
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {action.shape}")
        if np.any(action < 0) or np.any(action > 2):
            raise ValueError("Each action component must be in {0, 1, 2}")

        delta_p = self._action_to_delta_p(action)
        self.pressure = np.clip(self.pressure + delta_p, 0.0, p.P_max)
        self.last_action = action.copy()

        dt_sub = p.dt / p.n_substeps
        y = self.state.copy()
        t_local = self.t
        for _ in range(p.n_substeps):
            y = self._rk4_step(y, self.pressure, t_local, dt_sub)
            t_local += dt_sub

        self.state = y
        self.t = t_local
        self.step_count += 1

        reward = float(self.compute_reward())
        max_speed = float(np.max(np.abs(self.state[2:])))
        terminated = bool(max_speed > p.v_fail)
        truncated = bool(self.t >= p.t_max)
        if p.max_episode_steps is not None:
            truncated = truncated or (self.step_count >= p.max_episode_steps)

        obs = self._get_obs()
        info = self._get_info()
        info["action_raw"] = action.tolist()
        info["delta_p"] = delta_p.tolist()
        info["max_speed"] = max_speed
        info["terminated_reason"] = "v_fail" if terminated else None
        info["truncated_reason"] = "t_max" if (not terminated and self.t >= p.t_max) else None
        return obs, reward, terminated, truncated, info

    def render(self):
        info = self._get_info()
        print(
            f"t={info['t']:.4f}, x=({info['x1']:.4f}, {info['x2']:.4f}), "
            f"v=({info['v1']:.4f}, {info['v2']:.4f}), "
            f"P=({info['P1']:.4f}, {info['P2']:.4f}), "
            f"U={info['potential_energy']:.4f}, K={info['kinetic_energy']:.4f}"
        )

    def close(self):
        pass


# ---------- rollout helpers ----------
def run_no_control_episode(
    env: TwoBlockPressureEnv,
    *,
    seed: Optional[int] = None,
    reset_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Run one baseline episode with neutral action [1, 1]."""
    obs, info = env.reset(seed=seed, options=reset_options)
    history = {
        "obs": [obs.copy()],
        "reward": [],
        "t": [info["t"]],
        "x1": [info["x1"]],
        "x2": [info["x2"]],
        "v1": [info["v1"]],
        "v2": [info["v2"]],
        "P1": [info["P1"]],
        "P2": [info["P2"]],
        "U": [info["potential_energy"]],
        "K": [info["kinetic_energy"]],
    }
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(np.array([1, 1], dtype=np.int64))
        history["obs"].append(obs.copy())
        history["reward"].append(reward)
        history["t"].append(info["t"])
        history["x1"].append(info["x1"])
        history["x2"].append(info["x2"])
        history["v1"].append(info["v1"])
        history["v2"].append(info["v2"])
        history["P1"].append(info["P1"])
        history["P2"].append(info["P2"])
        history["U"].append(info["potential_energy"])
        history["K"].append(info["kinetic_energy"])
        done = truncated
    return {k: np.asarray(v) for k, v in history.items()}


def run_policy_episode(
    env: TwoBlockPressureEnv,
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    seed: Optional[int] = None,
    reset_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    obs, info = env.reset(seed=seed, options=reset_options)
    history = {
        "obs": [obs.copy()],
        "reward": [],
        "action": [],
        "t": [info["t"]],
        "x1": [info["x1"]],
        "x2": [info["x2"]],
        "v1": [info["v1"]],
        "v2": [info["v2"]],
        "P1": [info["P1"]],
        "P2": [info["P2"]],
        "U": [info["potential_energy"]],
        "K": [info["kinetic_energy"]],
    }
    done = False
    while not done:
        action = np.asarray(policy(obs), dtype=np.int64)
        obs, reward, terminated, truncated, info = env.step(action)
        history["obs"].append(obs.copy())
        history["reward"].append(reward)
        history["action"].append(action.copy())
        history["t"].append(info["t"])
        history["x1"].append(info["x1"])
        history["x2"].append(info["x2"])
        history["v1"].append(info["v1"])
        history["v2"].append(info["v2"])
        history["P1"].append(info["P1"])
        history["P2"].append(info["P2"])
        history["U"].append(info["potential_energy"])
        history["K"].append(info["kinetic_energy"])
        done = terminated or truncated
    return {k: np.asarray(v) for k, v in history.items()}


def simulate_with_solve_ivp(
    params: Optional[TwoBlockPressureParams] = None,
    *,
    t_span: Optional[Tuple[float, float]] = None,
    t_eval: Optional[np.ndarray] = None,
    y0: Optional[np.ndarray] = None,
    pressure: Tuple[float, float] = (0.0, 0.0),
    method: str = "Radau",
    rtol: float = 1e-7,
    atol: float = 1e-9,
):
    """Helper to reproduce the original solve_ivp-style baseline or constant-pressure runs."""
    if solve_ivp is None:
        raise ImportError("scipy is required for simulate_with_solve_ivp().")

    p = params or TwoBlockPressureParams()
    env = TwoBlockPressureEnv(p)
    t_span = t_span or (p.t0, p.t_max)
    if t_eval is None:
        n = max(2, int((t_span[1] - t_span[0]) / p.dt) + 1)
        t_eval = np.linspace(t_span[0], t_span[1], n)
    if y0 is None:
        y0 = np.array([p.x1_0, p.x2_0, p.v1_0, p.v2_0], dtype=np.float64)

    pressure_arr = np.array(pressure, dtype=np.float64)
    fun = lambda t, y: env.rhs(t, y, pressure_arr)
    return solve_ivp(fun=fun, t_span=t_span, y0=y0, t_eval=t_eval, method=method, rtol=rtol, atol=atol)


def example_heuristic_policy(obs: np.ndarray) -> np.ndarray:
    """Simple hand-crafted policy for smoke testing.

    - increase pressure on both blocks if both speeds are small and time is later,
    - decrease pressure if either block is moving too fast,
    - otherwise hold pressure.
    """
    v1, v2 = obs[2], obs[3]
    t_norm = obs[8]
    max_v = max(abs(v1), abs(v2))
    if max_v > 5.0:
        return np.array([2, 2], dtype=np.int64)
    if max_v < 0.5 and t_norm > 1.0:
        return np.array([0, 0], dtype=np.int64)
    return np.array([1, 1], dtype=np.int64)


if __name__ == "__main__":
    params = TwoBlockPressureParams(
        A=10.0,
        B=1.0,
        C=83.2,
        mu_s=0.6,
        mu_k=0.5,
        eps=1e-6,
        pressure_scale=1.0,
        P_max=0.95,
        dP=0.01,
        dt=1e-3,
        n_substeps=10,
        t_max=5.0,
        v_fail=20.0,
        w_U=1.0,
        w_K=1.0,
        U_ref=1.0,
        K_ref=1.0,
        x_ref=1.0,
        v_ref=1.0,
        t_ref=1.0,
        x1_0=0.0,
        x2_0=0.1,
        v1_0=0.0,
        v2_0=0.0,
    )

    env = TwoBlockPressureEnv(params=params, seed=0)
    baseline = run_no_control_episode(env, seed=0)
    print(
        f"No-control rollout finished at t={baseline['t'][-1]:.4f}, "
        f"max|v|={max(np.max(np.abs(baseline['v1'])), np.max(np.abs(baseline['v2']))):.4f}"
    )

    controlled = run_policy_episode(env, example_heuristic_policy, seed=1)
    print(
        f"Controlled rollout finished at t={controlled['t'][-1]:.4f}, "
        f"max|v|={max(np.max(np.abs(controlled['v1'])), np.max(np.abs(controlled['v2']))):.4f}"
    )

    print("Parameters:")
    print(asdict(params))
