import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.render_metadata import write_render_metadata
from src.render_viewer import show_render_gallery


PROJECT_ROOT = Path(__file__).resolve().parent


def plot_orbit(csv_path=None, output_path=None, show=False):
    """Plot a simulated XYZ trajectory and save it as a PNG image."""
    source = Path(csv_path) if csv_path else PROJECT_ROOT / "orbit_data.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Orbit CSV not found: {source}. Run orbit_simulation.py first.")
    with source.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows or not {"X", "Y", "Z"}.issubset(rows[0]):
        raise ValueError("Orbit CSV must contain X, Y, and Z columns")

    x_values = [float(row["X"]) for row in rows]
    y_values = [float(row["Y"]) for row in rows]
    z_values = [float(row["Z"]) for row in rows]
    destination = Path(output_path) if output_path else PROJECT_ROOT / "outputs" / "renders" / "orbit.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(8, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(x_values, y_values, z_values, color="#24a88a", linewidth=1.6, label="Star orbit")
    axis.scatter([0], [0], [0], color="#f08b42", s=90, label="Sagittarius A*")
    axis.scatter([x_values[0]], [y_values[0]], [z_values[0]], color="#3377bb", s=28, label="Start")
    axis.set(title="Stellar orbit around Sagittarius A*", xlabel="X (pc)", ylabel="Y (pc)", zlabel="Z (pc)")
    axis.legend()
    figure.savefig(destination, dpi=160)
    if show:
        plt.show()
    plt.close(figure)
    write_render_metadata(destination, {
        "render_type": "3D orbit plot",
        "data_source": str(source.resolve()),
        "source_kind": "simulated orbit CSV",
        "columns_used": ["X", "Y", "Z"],
        "coordinate_units": "parsecs (as defined by orbit_simulation.py)",
        "code_modules": ["plot_orbit.py", "orbit_simulation.py"],
        "plot_dimensions": [8, 7],
    })
    print(f"Saved orbit plot to {destination}")
    if show:
        show_render_gallery([destination])
    return destination


def main():
    parser = argparse.ArgumentParser(description="Plot orbit_data.csv as a 3D path.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    plot_orbit(args.input, args.output, args.show)


if __name__ == "__main__":
    main()