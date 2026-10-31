import numpy as np
import pytest

from src.model import (
    PARAMS, check_parameters, physical_parameters,
    solve_motion, energy, period_from_solution, approximate_period,
)


def test_parameters_ok():
    check_parameters(PARAMS)


def test_moment_of_inertia():
    I, mass, d, w0 = physical_parameters(PARAMS)
    L = PARAMS["rod_length"]
    r = PARAMS["rod_radius"]
    R = PARAMS["ball_radius"]
    m1 = PARAMS["rod_mass"]
    m2 = PARAMS["ball_mass"]
    expected = m1 * (L**2 / 3 + r**2 / 4) + m2 * (2 * R**2 / 5 + (L + R)**2)
    assert I == pytest.approx(expected)
    assert mass == pytest.approx(m1 + m2)
    assert d > 0
    assert w0 > 0


@pytest.mark.parametrize("angle", [5, 15, 30])
def test_period(angle):
    result = solve_motion(PARAMS, angle_deg=angle, friction=0)
    assert period_from_solution(result) == pytest.approx(
        approximate_period(PARAMS, angle), rel=5e-4
    )


@pytest.mark.parametrize("angle", [60, 90, 120])
def test_large_angle_period(angle):
    I, M, d, w0 = physical_parameters(PARAMS)
    result = solve_motion(PARAMS, angle_deg=angle, friction=0)
    assert period_from_solution(result) > 2 * np.pi / w0


def test_period_increases():
    angles = [5, 15, 30, 60, 90, 120]
    periods = [period_from_solution(solve_motion(PARAMS, angle_deg=angle, friction=0)) for angle in angles]
    assert periods == sorted(periods)


def test_energy_without_friction():
    result = solve_motion(PARAMS, friction=0)
    e = energy(PARAMS, result)
    assert np.max(np.abs(e / e[0] - 1)) < 1e-6


def test_energy_with_friction():
    result = solve_motion(PARAMS)
    e = energy(PARAMS, result)
    assert e[-1] < e[0]
    assert np.max(np.diff(e)) < 1e-7


def test_small_angles():
    I, M, d, w0 = physical_parameters(PARAMS)
    assert approximate_period(PARAMS, 1.0) == pytest.approx(2 * np.pi / w0, rel=1e-4)


@pytest.mark.parametrize("field,value", [
    ("rod_length", -1),
    ("ball_radius", 0.5),
    ("rod_mass", 1000),
    ("angle_deg", 190),
    ("angle_deg", 0),
    ("friction", -0.01),
    ("friction", 10),
    ("g", float("nan")),
    ("g", float("inf")),
    ("max_step", 1.0),
    ("rod_radius", "не число"),
    ("rod_radius", None),
    ("rod_length", True),
    ("angle_deg", [10]),
])
def test_wrong_input(field, value):
    p = PARAMS.copy()
    p[field] = value
    with pytest.raises(ValueError):
        check_parameters(p)


def test_missing_input():
    p = PARAMS.copy()
    del p["g"]
    with pytest.raises(ValueError):
        check_parameters(p)


def test_rod_density():
    p = PARAMS.copy()
    p["rod_mass"] = 0.02
    with pytest.raises(ValueError):
        check_parameters(p)


def test_ball_density():
    p = PARAMS.copy()
    p["ball_radius"] = 0.02
    with pytest.raises(ValueError):
        check_parameters(p)


def test_rod_thicker_than_ball():
    p = PARAMS.copy()
    p["rod_radius"] = 0.03
    p["ball_radius"] = 0.02
    with pytest.raises(ValueError):
        check_parameters(p)
