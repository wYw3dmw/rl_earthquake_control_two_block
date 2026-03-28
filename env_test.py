from two_block_pressure_env import (
    TwoBlockPressureParams,
    TwoBlockPressureEnv,
    run_no_control_episode,
    run_policy_episode,
    simulate_with_solve_ivp,
    example_heuristic_policy,
)
import numpy as np
import matplotlib.pyplot as plt

params = TwoBlockPressureParams()
env = TwoBlockPressureEnv(params)

traj_radau = simulate_with_solve_ivp(params)
traj_env = run_no_control_episode(env)
traj_heuristic = run_policy_episode(env, example_heuristic_policy)

t_env = traj_env["t"]
t_heuristic = traj_heuristic["t"]
t_radau = traj_radau.t

x1_radau = traj_radau.y[0]
x2_radau = traj_radau.y[1]
v1_radau = traj_radau.y[2]
v2_radau = traj_radau.y[3]

x1_radau_interp = np.interp(t_env, t_radau, x1_radau)
x2_radau_interp = np.interp(t_env, t_radau, x2_radau)
v1_radau_interp = np.interp(t_env, t_radau, v1_radau)
v2_radau_interp = np.interp(t_env, t_radau, v2_radau)

x1_heuristic_interp = np.interp(t_env, t_heuristic, traj_heuristic["x1"])
x2_heuristic_interp = np.interp(t_env, t_heuristic, traj_heuristic["x2"])
v1_heuristic_interp = np.interp(t_env, t_heuristic, traj_heuristic["v1"])
v2_heuristic_interp = np.interp(t_env, t_heuristic, traj_heuristic["v2"])

mask_after_heuristic = t_env > t_heuristic[-1]
x1_heuristic_interp[mask_after_heuristic] = np.nan
x2_heuristic_interp[mask_after_heuristic] = np.nan
v1_heuristic_interp[mask_after_heuristic] = np.nan
v2_heuristic_interp[mask_after_heuristic] = np.nan

plt.figure()
plt.plot(t_env, traj_env["x1"], label="x1 RK4")
plt.plot(t_env, x1_radau_interp, "--", label="x1 Radau")
plt.plot(t_env, x1_heuristic_interp, ":", label="x1 Heuristic")
plt.plot(t_env, traj_env["x2"], label="x2 RK4")
plt.plot(t_env, x2_radau_interp, "--", label="x2 Radau")
plt.plot(t_env, x2_heuristic_interp, ":", label="x2 Heuristic")
plt.legend()
plt.title("Slip comparison")
plt.grid()

plt.figure()
plt.plot(t_env, traj_env["v1"], label="v1 RK4")
plt.plot(t_env, v1_radau_interp, "--", label="v1 Radau")
plt.plot(t_env, v1_heuristic_interp, ":", label="v1 Heuristic")
plt.plot(t_env, traj_env["v2"], label="v2 RK4")
plt.plot(t_env, v2_radau_interp, "--", label="v2 Radau")
plt.plot(t_env, v2_heuristic_interp, ":", label="v2 Heuristic")
plt.legend()
plt.title("Slip rate comparison")
plt.grid()

KE_env = 0.5 * (traj_env["v1"]**2 + traj_env["v2"]**2)
KE_radau = 0.5 * (v1_radau_interp**2 + v2_radau_interp**2)
KE_heuristic = 0.5 * (v1_heuristic_interp**2 + v2_heuristic_interp**2)

plt.figure()
plt.plot(t_env, KE_env, label="RK4")
plt.plot(t_env, KE_radau, "--", label="Radau")
plt.plot(t_env, KE_heuristic, ":", label="Heuristic")
plt.legend()
plt.title("Kinetic Energy")
plt.grid()

X = 10 * t_env + 10

PE_env = 0.5 * params.A * (X - traj_env["x1"])**2 \
       + 0.5 * params.A * (X - traj_env["x2"])**2 \
       + 0.5 * params.B * (traj_env["x1"] - traj_env["x2"])**2

PE_radau = 0.5 * params.A * (X - x1_radau_interp)**2 \
         + 0.5 * params.A * (X - x2_radau_interp)**2 \
         + 0.5 * params.B * (x1_radau_interp - x2_radau_interp)**2

PE_heuristic = 0.5 * params.A * (X - x1_heuristic_interp)**2 \
         + 0.5 * params.A * (X - x2_heuristic_interp)**2 \
         + 0.5 * params.B * (x1_heuristic_interp - x2_heuristic_interp)**2

plt.figure()
plt.plot(t_env, PE_env, label="RK4")
plt.plot(t_env, PE_radau, "--", label="Radau")
plt.plot(t_env, PE_heuristic, ":", label="Heuristic")
plt.legend()
plt.title("Potential Energy")
plt.grid()

error_v1 = traj_env["v1"] - v1_radau_interp

plt.figure()
plt.plot(t_env, error_v1)
plt.title("Velocity error v1 (RK4 - Radau)")
plt.grid()

plt.show()