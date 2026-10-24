import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
GRAVITATIONAL_CONSTANT = 4.30091e-3
BLACK_HOLE_MASS = 4.0e6
KM_S_TO_PC_MYR = 1.022712165


def _acceleration(position):
    radius = float(np.linalg.norm(position))
    if radius == 0.0:
        raise ValueError("The simulated star reached the black-hole position")
    return -GRAVITATIONAL_CONSTANT * BLACK_HOLE_MASS * position / radius**3 * KM_S_TO_PC_MYR


def run_simulation(output_path=None, steps=1000, dt_myr=1e-5):
    """Integrate a 3D stellar orbit with velocity-Verlet and save CSV samples."""
    if steps < 1 or dt_myr <= 0:
        raise ValueError("steps and dt_myr must be positive")
    destination = Path(output_path) if output_path else PROJECT_ROOT / "orbit_data.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    position = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    velocity = np.array([0.0, 400.0, 150.0], dtype=np.float64)
    acceleration = _acceleration(position)

    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Step", "Time_Myr", "X", "Y", "Z"])
        for step in range(steps):
            writer.writerow([step, step * dt_myr, *position.tolist()])
            half_velocity = velocity + 0.5 * dt_myr * acceleration
            position = position + dt_myr * KM_S_TO_PC_MYR * half_velocity
            next_acceleration = _acceleration(position)
            velocity = half_velocity + 0.5 * dt_myr * next_acceleration
            acceleration = next_acceleration

    print(f"Saved {steps} orbit samples to {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description="Simulate a 3D orbit around Sagittarius A*.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt-myr", type=float, default=1e-5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_simulation(args.output, args.steps, args.dt_myr)


if __name__ == "__main__":
    main()