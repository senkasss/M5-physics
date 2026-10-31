from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import (
    PARAMS, check_parameters, physical_parameters, solve_motion,
    energy, period_from_solution, approximate_period,
)


def main():
    check_parameters(PARAMS)

    output = Path(__file__).resolve().parents[1] / "docs" / "plots"
    output.mkdir(parents=True, exist_ok=True)

    I, M, d, w0 = physical_parameters(PARAMS)
    free = solve_motion(PARAMS, friction=0)
    damped = solve_motion(PARAMS)

    plt.figure()
    plt.plot(free.t, np.rad2deg(free.y[0]), label="Без трения")
    plt.plot(damped.t, np.rad2deg(damped.y[0]), label="С трением")
    plt.xlabel("Время, с")
    plt.ylabel("Угол, градусы")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output / "01_oscillations.png")
    plt.close()

    angles = [5, 15, 30, 60, 90, 120]
    periods = []
    for angle in angles:
        solution = solve_motion(PARAMS, angle_deg=angle, friction=0)
        periods.append(period_from_solution(solution))

    small_angles = [5, 15, 30]
    theory = [approximate_period(PARAMS, angle) for angle in small_angles]

    plt.figure(figsize=(8, 4.5))
    plt.plot(angles, periods, "o-", label="Расчёт")
    plt.plot(small_angles, theory, "--", label="Приближение до 30°")
    plt.axhline(2 * np.pi / w0, linestyle="--", label="Малые углы")
    plt.xlabel("Начальный угол, градусы")
    plt.ylabel("Период, с")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output / "02_period.png", dpi=160)
    plt.close()

    E1 = energy(PARAMS, free)
    E2 = energy(PARAMS, damped)

    plt.figure()
    plt.plot(free.t, E1 / E1[0], label="Без трения")
    plt.plot(damped.t, E2 / E2[0], label="С трением")
    plt.xlabel("Время, с")
    plt.ylabel("E / E(0)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output / "03_energy.png")
    plt.close()

    small_angle = 8
    small = solve_motion(PARAMS, angle_deg=small_angle)
    beta = PARAMS["friction"] / (2 * I)
    upper = small_angle * np.exp(-beta * small.t)

    plt.figure()
    plt.plot(small.t, np.rad2deg(small.y[0]), label="Расчёт")
    plt.plot(small.t, upper, "--", label="Огибающая")
    plt.plot(small.t, -upper, "--")
    plt.xlabel("Время, с")
    plt.ylabel("Угол, градусы")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output / "04_damping.png")
    plt.close()

    print("Момент инерции:", round(I, 6), "кг*м^2")
    print("Центр масс:", round(d, 4), "м")
    print("Период малых колебаний:", round(2 * np.pi / w0, 4), "с")
    print("Ошибка энергии без трения:", f"{np.max(np.abs(E1 / E1[0] - 1)):.2e}")
    print("Энергия с трением в конце:", f"{E2[-1] / E2[0]:.1%}")
    print("Угол | Расчёт | Приближение")
    for angle, t1 in zip(angles, periods):
        t2 = f"{approximate_period(PARAMS, angle):.4f} с" if angle <= 30 else "—"
        print(f"{angle:>3}° | {t1:.4f} с | {t2}")
    print("Графики:", output)


main()
