from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from two_block_pressure_env import TwoBlockPressureEnv, TwoBlockPressureParams


def run_episode_for_compare(
    env: TwoBlockPressureEnv,
    action_fn,
    *,
    seed: int | None = None,
    reset_options: dict | None = None,
    ignore_termination: bool = True,
) -> dict[str, np.ndarray]:
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
        "max_speed": [max(abs(info["v1"]), abs(info["v2"]))],
        "terminated_flag": [False],
        "truncated_flag": [False],
    }

    done = False
    while not done:
        action = np.asarray(action_fn(obs), dtype=np.int64)
        if action.shape != (2,):
            raise ValueError(f"action_fn must return shape (2,), got {action.shape}")

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
        history["max_speed"].append(info["max_speed"])
        history["terminated_flag"].append(terminated)
        history["truncated_flag"].append(truncated)

        if ignore_termination:
            done = truncated
        else:
            done = terminated or truncated

    return {k: np.asarray(v) for k, v in history.items()}


def plot_two_block_compare(
    ax,
    t_base,
    y1_base,
    y2_base,
    t_ctrl,
    y1_ctrl,
    y2_ctrl,
    ylabel: str,
    title: str,
):
    ax.plot(t_base, y1_base, label="No control - block 1")
    ax.plot(t_base, y2_base, label="No control - block 2")
    ax.plot(t_ctrl, y1_ctrl, label="Policy - block 1")
    ax.plot(t_ctrl, y2_ctrl, label="Policy - block 2")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def plot_scalar_compare(
    ax,
    t_base,
    y_base,
    t_ctrl,
    y_ctrl,
    ylabel: str,
    title: str,
):
    ax.plot(t_base, y_base, label="No control")
    ax.plot(t_ctrl, y_ctrl, label="Policy")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def save_single_figures(baseline: dict[str, np.ndarray],
                        controlled: dict[str, np.ndarray],
                        outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_two_block_compare(
        ax,
        baseline["t"], baseline["x1"], baseline["x2"],
        controlled["t"], controlled["x1"], controlled["x2"],
        ylabel="Slip",
        title="Slip comparison",
    )
    fig.tight_layout()
    fig.savefig(outdir / "compare_slip.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_two_block_compare(
        ax,
        baseline["t"], baseline["v1"], baseline["v2"],
        controlled["t"], controlled["v1"], controlled["v2"],
        ylabel="Slip rate",
        title="Slip rate comparison",
    )
    fig.tight_layout()
    fig.savefig(outdir / "compare_slip_rate.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_scalar_compare(
        ax,
        baseline["t"], baseline["K"],
        controlled["t"], controlled["K"],
        ylabel="Kinetic energy",
        title="Kinetic energy comparison",
    )
    fig.tight_layout()
    fig.savefig(outdir / "compare_kinetic_energy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_scalar_compare(
        ax,
        baseline["t"], baseline["U"],
        controlled["t"], controlled["U"],
        ylabel="Potential energy",
        title="Potential energy comparison",
    )
    fig.tight_layout()
    fig.savefig(outdir / "compare_potential_energy.png", dpi=200)
    plt.close(fig)


def save_summary_figure(baseline: dict[str, np.ndarray],
                        controlled: dict[str, np.ndarray],
                        outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_two_block_compare(
        axes[0, 0],
        baseline["t"], baseline["x1"], baseline["x2"],
        controlled["t"], controlled["x1"], controlled["x2"],
        ylabel="Slip",
        title="Slip comparison",
    )

    plot_two_block_compare(
        axes[0, 1],
        baseline["t"], baseline["v1"], baseline["v2"],
        controlled["t"], controlled["v1"], controlled["v2"],
        ylabel="Slip rate",
        title="Slip rate comparison",
    )

    plot_scalar_compare(
        axes[1, 0],
        baseline["t"], baseline["K"],
        controlled["t"], controlled["K"],
        ylabel="Kinetic energy",
        title="Kinetic energy comparison",
    )

    plot_scalar_compare(
        axes[1, 1],
        baseline["t"], baseline["U"],
        controlled["t"], controlled["U"],
        ylabel="Potential energy",
        title="Potential energy comparison",
    )

    fig.tight_layout()
    fig.savefig(outdir / "compare_summary_2x2.png", dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained SB3 PPO model zip file.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="compare_results",
        help="Directory to save comparison figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for both rollouts.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    outdir = Path(args.outdir)
    seed = args.seed

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    outdir.mkdir(parents=True, exist_ok=True)

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
        t_max=7.0,
        v_fail=20.0,
        x_ref=1.0,
        v_ref=1.0,
        t_ref=1.0,
        x1_0=0.0,
        x2_0=0.1,
        v1_0=0.0,
        v2_0=0.0,
    )

    env_base = TwoBlockPressureEnv(params=params, seed=seed)
    env_ctrl = TwoBlockPressureEnv(params=params, seed=seed)

    model = PPO.load(model_path)

    def no_control_action_fn(obs: np.ndarray) -> np.ndarray:
        return np.array([1, 1], dtype=np.int64)

    def policy_action_fn(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.int64)

    baseline = run_episode_for_compare(
        env_base,
        no_control_action_fn,
        seed=seed,
        ignore_termination=True,
    )

    controlled = run_episode_for_compare(
        env_ctrl,
        policy_action_fn,
        seed=seed,
        ignore_termination=True,
    )

    base_max_v = max(np.max(np.abs(baseline["v1"])), np.max(np.abs(baseline["v2"])))
    ctrl_max_v = max(np.max(np.abs(controlled["v1"])), np.max(np.abs(controlled["v2"])))

    print(f"No control: t_end = {baseline['t'][-1]:.4f}, max|v| = {base_max_v:.4f}")
    print(f"Policy:     t_end = {controlled['t'][-1]:.4f}, max|v| = {ctrl_max_v:.4f}")

    save_single_figures(baseline, controlled, outdir)
    save_summary_figure(baseline, controlled, outdir)

    print(f"Figures saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()