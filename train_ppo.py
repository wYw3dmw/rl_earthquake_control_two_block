from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from two_block_pressure_env import TwoBlockPressureEnv, TwoBlockPressureParams


class SoftFailWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        mode: str = "ignore",           # "ignore", "relax", "original"
        relaxed_factor: float = 1.5,
        fail_penalty: float = 0.0,
        obs_velocity_clip: float | None = None,
    ):
        super().__init__(env)
        self.mode = mode
        self.relaxed_factor = relaxed_factor
        self.fail_penalty = fail_penalty
        self.obs_velocity_clip = obs_velocity_clip

    def _clip_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_velocity_clip is None:
            return obs
        obs = np.array(obs, copy=True)
        obs[2] = np.clip(obs[2], -self.obs_velocity_clip, self.obs_velocity_clip)
        obs[3] = np.clip(obs[3], -self.obs_velocity_clip, self.obs_velocity_clip)
        obs[7] = np.clip(obs[7], -2.0 * self.obs_velocity_clip, 2.0 * self.obs_velocity_clip)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._clip_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        max_speed = info.get("max_speed", None)
        raw_failed = (info.get("terminated_reason") == "v_fail")
        info["raw_v_fail"] = raw_failed

        if raw_failed:
            reward -= self.fail_penalty

            if self.mode == "ignore":
                terminated = False

            elif self.mode == "relax":
                if max_speed is not None:
                    relaxed_fail = self.env.params.v_fail * self.relaxed_factor
                    if max_speed <= relaxed_fail:
                        terminated = False

            elif self.mode == "original":
                pass
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        if max_speed is not None and max_speed > 5.0 * self.env.params.v_fail:
            terminated = True
            info["terminated_reason"] = "hard_safety_stop"

        obs = self._clip_obs(obs)
        return obs, reward, terminated, truncated, info


class FixedResetOptionsWrapper(gym.Wrapper):
    """
    Always reset the env from a fixed set of initial conditions.
    """
    def __init__(self, env: gym.Env, reset_options: dict):
        super().__init__(env)
        self.fixed_reset_options = dict(reset_options)

    def reset(self, **kwargs):
        options = kwargs.pop("options", None) or {}
        merged_options = dict(self.fixed_reset_options)
        merged_options.update(options)
        return self.env.reset(options=merged_options, **kwargs)


def build_params() -> TwoBlockPressureParams:
    return TwoBlockPressureParams(
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
        n_substeps=20,
        t_max=7.0,
        v_fail=20.0,

        reward_av=2.0,
        reward_ax=2.0,
        reward_bv=0.1,
        reward_bx=0.05,

        # old reward params kept but unused by current reward
        w_U=1.0,
        w_K=2.0,
        U_ref=400.0,
        K_ref=300.0,
        reward_power_U=1.0,
        reward_power_K=2.0,

        # new reward params
        reward_eps=1e-8,
        reward_clip=50.0,

        x_ref=1.0,
        v_ref=1.0,
        t_ref=1.0,
        x1_0=0.0,
        x2_0=0.1,
        v1_0=0.0,
        v2_0=0.0,
        P1_0=0.0,
        P2_0=0.0,
        t0=0.0,
    )


def build_fixed_start_options(params: TwoBlockPressureParams, target_t: float = 1.5) -> dict:
    """
    Deterministically integrate the no-control system from t=0 to target_t.
    This state will be used as the reset state for all training/eval episodes.
    """
    env = TwoBlockPressureEnv(params=params, seed=0)
    env.reset()

    warm_action = np.array([1, 1], dtype=np.int64)  # neutral action => delta_p = [0, 0]
    delta_p = env._action_to_delta_p(warm_action)
    env.pressure = np.clip(env.pressure + delta_p, 0.0, env.params.P_max)
    env.last_action = warm_action.copy()

    y = env.state.copy()
    t_local = env.t

    while t_local < target_t:
        dt_step = min(env.params.dt, target_t - t_local)
        dt_sub = dt_step / env.params.n_substeps

        for _ in range(env.params.n_substeps):
            y = env._rk4_step(y, env.pressure, t_local, dt_sub)
            t_local += dt_sub

    env.state = y
    env.t = t_local

    return {
        "x1_0": float(env.state[0]),
        "x2_0": float(env.state[1]),
        "v1_0": float(env.state[2]),
        "v2_0": float(env.state[3]),
        "P1_0": float(env.pressure[0]),
        "P2_0": float(env.pressure[1]),
        "t0": float(env.t),
    }


def make_env(
    params: TwoBlockPressureParams,
    seed: int,
    *,
    training: bool,
    reset_options: dict | None = None,
    softfail_mode: str = "original",
    relaxed_factor: float = 1.5,
    fail_penalty: float = 2.0,
    obs_velocity_clip: float | None = 30.0,
) -> Callable[[], gym.Env]:
    def _factory():
        env = TwoBlockPressureEnv(params=params, seed=seed)

        if training:
            env = SoftFailWrapper(
                env,
                mode=softfail_mode,
                relaxed_factor=relaxed_factor,
                fail_penalty=fail_penalty,
                obs_velocity_clip=obs_velocity_clip,
            )

        if reset_options is not None:
            env = FixedResetOptionsWrapper(env, reset_options=reset_options)

        env = Monitor(env)
        return env

    return _factory


def build_train_env(
    params: TwoBlockPressureParams,
    seed: int,
    n_envs: int,
    *,
    reset_options: dict | None,
    softfail_mode: str,
    relaxed_factor: float = 1.5,
    fail_penalty: float = 2.0,
    obs_velocity_clip: float | None = 30.0,
):
    return DummyVecEnv([
        make_env(
            params=params,
            seed=seed + i,
            training=True,
            reset_options=reset_options,
            softfail_mode=softfail_mode,
            relaxed_factor=relaxed_factor,
            fail_penalty=fail_penalty,
            obs_velocity_clip=obs_velocity_clip,
        )
        for i in range(n_envs)
    ])


def evaluate_policy_once(
    model: PPO,
    env: TwoBlockPressureEnv,
    reset_options: dict | None = None,
):
    obs, info = env.reset(options=reset_options)
    history = {
        "t": [info["t"]],
        "x1": [info["x1"]],
        "x2": [info["x2"]],
        "v1": [info["v1"]],
        "v2": [info["v2"]],
        "P1": [info["P1"]],
        "P2": [info["P2"]],
        "U": [info["potential_energy"]],
        "K": [info["kinetic_energy"]],
        "reward": [],
        "action": [],
    }

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        history["reward"].append(float(reward))
        history["action"].append(np.array(action, dtype=np.int64))
        history["t"].append(info["t"])
        history["x1"].append(info["x1"])
        history["x2"].append(info["x2"])
        history["v1"].append(info["v1"])
        history["v2"].append(info["v2"])
        history["P1"].append(info["P1"])
        history["P2"].append(info["P2"])
        history["U"].append(info["potential_energy"])
        history["K"].append(info["kinetic_energy"])

    return {k: np.asarray(v) for k, v in history.items()}


def evaluate_multiple_episodes(
    model: PPO,
    env: TwoBlockPressureEnv,
    n_episodes: int = 5,
    reset_options: dict | None = None,
):
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs, info = env.reset(options=reset_options)
        terminated = False
        truncated = False
        ep_reward = 0.0
        ep_len = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
    }


def save_rollout_plot(history: dict[str, np.ndarray], out_path: Path, reward_chunk_size: int = 20) -> None:
    t = history["t"]
    rewards = history["reward"]

    fig = plt.figure(figsize=(10, 15))

    ax1 = fig.add_subplot(5, 1, 1)
    ax1.plot(t, history["x1"], label="x1")
    ax1.plot(t, history["x2"], label="x2")
    ax1.set_title("Slip")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(5, 1, 2)
    ax2.plot(t, history["v1"], label="v1")
    ax2.plot(t, history["v2"], label="v2")
    ax2.set_title("Slip rate")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(5, 1, 3)
    ax3.plot(t, history["K"], label="Kinetic energy")
    ax3.plot(t, history["U"], label="Potential energy")
    ax3.set_title("Energy")
    ax3.grid(True)
    ax3.legend()

    ax4 = fig.add_subplot(5, 1, 4)
    ax4.plot(t, history["P1"], label="P1")
    ax4.plot(t, history["P2"], label="P2")
    ax4.set_title("Pressure")
    ax4.grid(True)
    ax4.legend()

    ax5 = fig.add_subplot(5, 1, 5)
    if len(rewards) > 0:
        reward_chunks = [
            np.sum(rewards[i:i + reward_chunk_size])
            for i in range(0, len(rewards), reward_chunk_size)
        ]
        reward_chunk_t = [
            t[min(i + reward_chunk_size, len(t) - 1)]
            for i in range(0, len(rewards), reward_chunk_size)
        ]
        ax5.plot(reward_chunk_t, reward_chunks, marker="o")
    ax5.set_title(f"Chunked reward (sum over every {reward_chunk_size} steps)")
    ax5.set_xlabel("Time")
    ax5.grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_eval_curves(eval_log_dir: Path, out_path: Path) -> None:
    eval_file = eval_log_dir / "evaluations.npz"
    if not eval_file.exists():
        print(f"No evaluation file found at: {eval_file}")
        return

    data = np.load(eval_file)
    timesteps = data["timesteps"]
    results = data["results"]
    ep_lengths = data["ep_lengths"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)
    mean_lengths = ep_lengths.mean(axis=1)

    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(timesteps, mean_rewards, marker="o", label="Mean eval reward")
    ax1.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        label="±1 std",
    )
    ax1.set_title("Evaluation reward during training")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Reward")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(timesteps, mean_lengths, marker="o", label="Mean eval episode length")
    ax2.set_title("Evaluation episode length during training")
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Episode length")
    ax2.grid(True)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="ppo_two_block_runs")
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--resume-from-best", action="store_true")
    parser.add_argument("--resume-model-path", type=str, default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    params = build_params()
    reset_options = build_fixed_start_options(params, target_t=1.2)

    print("Using fixed reset state at approximately t=1.2:")
    print(reset_options)

    phase1_steps = int(args.total_timesteps * 0.3)
    phase2_steps = int(args.total_timesteps * 0.4)
    phase3_steps = args.total_timesteps - phase1_steps - phase2_steps

    eval_env = DummyVecEnv([
        make_env(
            params=params,
            seed=args.seed + 10_000,
            training=False,
            reset_options=reset_options,
        )
    ])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(outdir / "best_model"),
        log_path=str(outdir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    # Phase 1
    train_env = build_train_env(
        params=params,
        seed=args.seed,
        n_envs=args.n_envs,
        reset_options=reset_options,
        softfail_mode="ignore",
        relaxed_factor=1.5,
        fail_penalty=2.0,
        obs_velocity_clip=30.0,
    )

    resume_path = None
    if args.resume_model_path is not None:
        resume_path = Path(args.resume_model_path)
    elif args.resume_from_best:
        resume_path = outdir / "best_model" / "best_model.zip"

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume model not found: {resume_path}")

        print("=" * 60)
        print(f"Resuming training from: {resume_path}")
        print("=" * 60)

        model = PPO.load(
            str(resume_path),
            env=train_env,
            device="auto",
        )

        model.seed = args.seed

    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(outdir / "tb"),
            device="auto",
        )

    print("=" * 60)
    print("Phase 1 / 3: softened termination = ignore")
    print(f"Steps: {phase1_steps}")
    print("Episodes start from fixed state near t=1.2")
    print("=" * 60)

    model.learn(
        total_timesteps=phase1_steps,
        callback=eval_callback,
        progress_bar=True,
        reset_num_timesteps=True,
    )

    # Phase 2
    print("=" * 60)
    print("Phase 2 / 3: softened termination = relax")
    print(f"Steps: {phase2_steps}")
    print("Episodes start from fixed state near t=1.2")
    print("=" * 60)

    train_env_phase2 = build_train_env(
        params=params,
        seed=args.seed + 1000,
        n_envs=args.n_envs,
        reset_options=reset_options,
        softfail_mode="relax",
        relaxed_factor=1.5,
        fail_penalty=2.0,
        obs_velocity_clip=25.0,
    )
    model.set_env(train_env_phase2)
    model.learn(
        total_timesteps=phase2_steps,
        callback=eval_callback,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    # Phase 3
    print("=" * 60)
    print("Phase 3 / 3: original termination restored")
    print(f"Steps: {phase3_steps}")
    print("Episodes start from fixed state near t=1.2")
    print("=" * 60)

    train_env_phase3 = build_train_env(
        params=params,
        seed=args.seed + 2000,
        n_envs=args.n_envs,
        reset_options=reset_options,
        softfail_mode="original",
        relaxed_factor=1.0,
        fail_penalty=0.0,
        obs_velocity_clip=None,
    )
    model.set_env(train_env_phase3)
    model.learn(
        total_timesteps=phase3_steps,
        callback=eval_callback,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    model.save(outdir / "ppo_two_block_final")

    save_eval_curves(outdir / "eval_logs", outdir / "eval_curves.png")

    # final rollout
    test_env = TwoBlockPressureEnv(params=params, seed=args.seed)
    history_final = evaluate_policy_once(model, test_env, reset_options=reset_options)
    save_rollout_plot(history_final, outdir / "ppo_rollout_final.png", reward_chunk_size=20)

    # best rollout
    best_model_path = outdir / "best_model" / "best_model.zip"
    if best_model_path.exists():
        best_model = PPO.load(best_model_path)
        best_test_env = TwoBlockPressureEnv(params=params, seed=args.seed)
        history_best = evaluate_policy_once(best_model, best_test_env, reset_options=reset_options)
        save_rollout_plot(history_best, outdir / "ppo_rollout_best.png", reward_chunk_size=20)
        print("Best-model rollout saved.")
    else:
        print("Best model not found, skipped best-model rollout.")

    # extra evaluation summary
    summary_env = TwoBlockPressureEnv(params=params, seed=args.seed)
    summary_stats = evaluate_multiple_episodes(
        model,
        summary_env,
        n_episodes=5,
        reset_options=reset_options,
    )
    print("Post-training evaluation summary:")
    print(summary_stats)

    print("Training finished.")
    print(f"Artifacts saved to: {outdir.resolve()}")
    print(f"Final rollout length: {len(history_final['t']) - 1}")
    print(
        "Final rollout max |v|: "
        f"{max(np.max(np.abs(history_final['v1'])), np.max(np.abs(history_final['v2']))):.4f}"
    )


if __name__ == "__main__":
    main()