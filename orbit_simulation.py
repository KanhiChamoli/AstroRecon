from pathlib import Path
import runpy


if __name__ == "__main__":
    simulation = Path(__file__).resolve().parent / "AstroRecon" / "AstroRecon" / "orbit_simulation.py"
    runpy.run_path(str(simulation), run_name="__main__")