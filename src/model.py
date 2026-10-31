import numpy as np
from scipy.integrate import solve_ivp


PARAMS = {
    "rod_length": 0.60,
    "rod_radius": 0.008,
    "rod_mass": 0.35,
    "ball_radius": 0.08,
    "ball_mass": 1.60,
    "g": 9.81,
    "angle_deg": 30.0,
    "friction": 0.045,
    "time": 45.0,
    "max_step": 0.02,
}


def physical_parameters(p):
    L = p["rod_length"]
    r = p["rod_radius"]
    m1 = p["rod_mass"]
    R = p["ball_radius"]
    m2 = p["ball_mass"]

    I1 = m1 * (L**2 / 3 + r**2 / 4)
    I2 = m2 * (2 * R**2 / 5 + (L + R)**2)
    I = I1 + I2
    M = m1 + m2
    d = (m1 * L / 2 + m2 * (L + R)) / M
    w0 = np.sqrt(M * p["g"] * d / I)
    return I, M, d, w0


def check_parameters(p):
    limits = {
        "rod_length": (0.15, 2.0),
        "rod_radius": (0.002, 0.03),
        "rod_mass": (0.02, 5.0),
        "ball_radius": (0.02, 0.20),
        "ball_mass": (0.05, 15.0),
        "g": (9.5, 10.1),
        "angle_deg": (1.0, 120.0),
        "friction": (0.0, 0.15),
        "time": (15.0, 120.0),
        "max_step": (0.001, 0.05),
    }

    for name, (minimum, maximum) in limits.items():
        if name not in p:
            raise ValueError(f"Нет параметра {name}")
        value = p[name]
        if isinstance(value, (bool, str)) or not np.isscalar(value):
            raise ValueError(f"{name}: нужно число")
        try:
            value = float(value)
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"{name}: нужно число") from None
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name}: значение должно быть от {minimum} до {maximum}")

    if p["rod_radius"] > p["ball_radius"]:
        raise ValueError("Радиус стержня больше радиуса шара")

    L, r, m1 = p["rod_length"], p["rod_radius"], p["rod_mass"]
    R, m2 = p["ball_radius"], p["ball_mass"]
    rod_density = m1 / (np.pi * r**2 * L)
    ball_density = m2 / (4 * np.pi * R**3 / 3)

    if not 400 <= rod_density <= 20000:
        raise ValueError("Неверная плотность стержня")
    if not 300 <= ball_density <= 22000:
        raise ValueError("Неверная плотность шара")

    I, M, d, w0 = physical_parameters(p)
    if p["friction"] / (2 * I * w0) > 0.05:
        raise ValueError("Слишком большое трение")


def solve_motion(p, angle_deg=None, friction=None):
    data = p.copy()
    if angle_deg is not None:
        data["angle_deg"] = angle_deg
    if friction is not None:
        data["friction"] = friction
    check_parameters(data)

    I, M, d, w0 = physical_parameters(data)

    def equations(t, y):
        angle, speed = y
        return [speed, -(M * data["g"] * d * np.sin(angle) + data["friction"] * speed) / I]

    def crossing(t, y):
        return y[0]

    crossing.direction = -1
    crossing.terminal = False

    result = solve_ivp(
        equations,
        (0, data["time"]),
        [np.deg2rad(data["angle_deg"]), 0],
        events=crossing,
        max_step=data["max_step"],
        rtol=1e-9,
        atol=1e-11,
    )
    return result


def energy(p, result):
    I, M, d, w0 = physical_parameters(p)
    angle, speed = result.y
    return I * speed**2 / 2 + M * p["g"] * d * (1 - np.cos(angle))


def period_from_solution(result):
    times = result.t_events[0]
    if len(times) < 2:
        raise ValueError("Слишком мало колебаний")
    return float(np.mean(np.diff(times)))


def approximate_period(p, angle_deg):
    I, M, d, w0 = physical_parameters(p)
    angle = np.deg2rad(angle_deg)
    return 2 * np.pi / w0 * (1 + angle**2 / 16)
