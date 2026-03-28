import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

params = {
    "A": 10.0,
    "B": 1.0,
    "C": 83.2,
    "mu_s": 0.6,
    "mu_k": 0.5,
    "v_tol": 1e-6,
}

def friction_coeff(x, mu_s, mu_k):
    s = max(x, 0.0)
    return mu_k + (mu_s - mu_k) * np.exp(-s)

def block_dynamics(Fd, v, Fc, v_tol):
    global stick_count, slip_count

    if abs(v) < v_tol and abs(Fd) <= Fc:
        stick_count += 1
        dxdt = 0.0
        dvdt = 0.0
        state = "stick"
    else:
        slip_count += 1

        if abs(v) >= v_tol:
            direction = np.sign(v)
        else:
            direction = np.sign(Fd)

        Ff = Fc * direction
        dxdt = v
        dvdt = Fd - Ff
        state = "slip"

    return dxdt, dvdt, state

def rhs(t, y, params):
    x1, x2, v1, v2 = y

    A = params["A"]
    B = params["B"]
    C = params["C"]
    mu_s = params["mu_s"]
    mu_k = params["mu_k"]
    v_tol = params["v_tol"]

    mu1 = friction_coeff(x1, mu_s, mu_k)
    mu2 = friction_coeff(x2, mu_s, mu_k)

    Fd1 = A * (10 * t + 10 - x1) + B * (x2 - x1)
    Fd2 = A * (10 * t + 10 - x2) + B * (x1 - x2)

    Fc1 = C * mu1
    Fc2 = C * mu2

    dx1dt, dv1dt, _ = block_dynamics(Fd1, v1, Fc1, v_tol)
    dx2dt, dv2dt, _ = block_dynamics(Fd2, v2, Fc2, v_tol)

    return [dx1dt, dx2dt, dv1dt, dv2dt]


y0 = [0.0, 0.1, 0.0, 0.0]

t_span = (0.0, 10.0)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

stick_count = 0
slip_count = 0

sol = solve_ivp(
    fun=lambda t, y: rhs(t, y, params),
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method="RK45",
    rtol=1e-7,
    atol=1e-9
)

print("success =", sol.success)
print("message =", sol.message)

total_count = stick_count + slip_count
print("stick_count =", stick_count)
print("slip_count  =", slip_count)
print("stick ratio =", stick_count / total_count if total_count > 0 else 0.0)

t = sol.t
x1, x2, v1, v2 = sol.y

A = params["A"]
B = params["B"]
C = params["C"]
mu_s = params["mu_s"]
mu_k = params["mu_k"]

a1 = np.zeros_like(t)
a2 = np.zeros_like(t)

for i in range(len(t)):
    s1 = max(x1[i], 0.0)
    s2 = max(x2[i], 0.0)
    mu1 = mu_k + (mu_s - mu_k) * np.exp(-s1)
    mu2 = mu_k + (mu_s - mu_k) * np.exp(-s2)

    Fd1 = A * (10 * t[i] + 10 - x1[i]) + B * (x2[i] - x1[i])
    Fd2 = A * (10 * t[i] + 10 - x2[i]) + B * (x1[i] - x2[i])

    Fc1 = C * mu1
    Fc2 = C * mu2

    if abs(v1[i]) < params["v_tol"] and abs(Fd1) <= Fc1:
        a1[i] = 0.0
    else:
        direction1 = np.sign(v1[i]) if abs(v1[i]) >= params["v_tol"] else np.sign(Fd1)
        Ff1 = Fc1 * direction1
        a1[i] = Fd1 - Ff1

    if abs(v2[i]) < params["v_tol"] and abs(Fd2) <= Fc2:
        a2[i] = 0.0
    else:
        direction2 = np.sign(v2[i]) if abs(v2[i]) >= params["v_tol"] else np.sign(Fd2)
        Ff2 = Fc2 * direction2
        a2[i] = Fd2 - Ff2

mu1 = np.array([friction_coeff(xx, mu_s, mu_k) for xx in x1])
mu2 = np.array([friction_coeff(xx, mu_s, mu_k) for xx in x2])

Fd1 = A * (10 * t + 10 - x1) + B * (x2 - x1)
Fd2 = A * (10 * t + 10 - x2) + B * (x1 - x2)

Fc1 = C * mu1
Fc2 = C * mu2

r1 = np.abs(Fd1) / Fc1
r2 = np.abs(Fd2) / Fc2

KE = 0.5 * (v1**2 + v2**2)
X = 10 * t + 10
PE = 0.5 * A * (X - x1)**2 + 0.5 * A * (X - x2)**2 + 0.5 * B * (x1 - x2)**2
TE = KE + PE

plt.figure(figsize=(8, 5))
plt.plot(t, x1, label="x1")
plt.plot(t, x2, label="x2")
plt.xlabel("t*")
plt.ylabel("x*")
plt.title("Slip vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, v1, label="v1")
plt.plot(t, v2, label="v2")
plt.xlabel("t*")
plt.ylabel("v*")
plt.title("Slip Rate vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, KE, label="KE")
plt.xlabel("t*")
plt.ylabel("K*")
plt.title("Kinetic Energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, PE, label="PE")
plt.xlabel("t*")
plt.ylabel("U*")
plt.title("Potential Energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, TE, label="TE")
plt.xlabel("t*")
plt.ylabel("E*")
plt.title("Total Mechanical Energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, r1, label="r1 = |Fd1|/Fc1")
plt.plot(t, r2, label="r2 = |Fd2|/Fc2")
plt.xlabel("t*")
plt.ylabel("r*")
plt.title("Driving Force / Friction Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, a1, label="a1")
plt.plot(t, a2, label="a2")
plt.xlabel("t*")
plt.ylabel("acceleration a*")
plt.title("Nondimensional Acceleration vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()