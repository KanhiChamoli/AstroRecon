from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform


DEPENDENCIES = ("astropy", "ehtim", "matplotlib", "numpy", "scipy", "torch")


def write_render_metadata(image_path, details):
	"""Write source, processing, and software provenance next to a rendered image."""
	image_path = Path(image_path)
	metadata_path = image_path.with_name(f"{image_path.stem}.metadata.json")
	package_versions = {}
	for package in DEPENDENCIES:
		try:
			package_versions[package] = version(package)
		except PackageNotFoundError:
			package_versions[package] = None

	metadata = {
		"artifact": image_path.name,
		"artifact_format": image_path.suffix.lstrip(".").lower(),
		"generated_utc": datetime.now(timezone.utc).isoformat(),
		"programming_language": "Python",
		"python_version": platform.python_version(),
		"libraries": package_versions,
		**details,
	}
	metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
	return metadata_path