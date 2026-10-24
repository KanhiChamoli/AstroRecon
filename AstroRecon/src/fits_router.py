"""Content-based identification of common astronomical FITS products."""

from pathlib import Path

from astropy.io import fits


def classify_fits(path):
	"""Return a description of a FITS file's primary scientific data product."""
	fits_path = Path(path)
	if not fits_path.is_file():
		raise FileNotFoundError(f"FITS input not found: {fits_path}")

	with fits.open(fits_path, memmap=False) as hdus:
		primary = hdus[0]
		header = primary.header
		extension_names = {
			str(hdu.header.get("EXTNAME", "")).strip().upper()
			for hdu in hdus[1:]
		}
		object_name = header.get("OBJECT")
		if object_name is None and "AIPS AN" in extension_names:
			object_name = hdus["AIPS AN"].header.get("OBJECT")

		is_random_groups = isinstance(primary, fits.GroupsHDU)
		axis_types = [
			str(header.get(f"CTYPE{axis}", "")).strip().upper()
			for axis in range(1, int(header.get("NAXIS", 0)) + 1)
		]
		is_uvfits = is_random_groups and "AIPS AN" in extension_names
		is_uvfits = is_uvfits or (
			"AIPS AN" in extension_names
			and "COMPLEX" in axis_types
			and any(axis in axis_types for axis in ("U", "UU", "UU---SIN"))
		)

		if is_uvfits:
			kind = "uvfits_visibility"
			data_shape = list(primary.data.shape) if primary.data is not None else None
			hdu_index = 0
		else:
			image_hdus = [
				(index, hdu)
				for index, hdu in enumerate(hdus)
				if getattr(hdu, "data", None) is not None
				and getattr(hdu.data, "ndim", 0) >= 2
				and not getattr(hdu.data.dtype, "names", None)
			]
			if image_hdus:
				hdu_index, image_hdu = image_hdus[0]
				data_shape = list(image_hdu.data.shape)
				axis_types = [
					str(image_hdu.header.get(f"CTYPE{axis}", "")).strip().upper()
					for axis in range(1, int(image_hdu.header.get("NAXIS", 0)) + 1)
				]
				if len(data_shape) == 2:
					kind = "image2d"
				elif len(data_shape) == 3:
					kind = "image_cube"
				else:
					kind = "image_hypercube"
				object_name = image_hdu.header.get("OBJECT", object_name)
			else:
				table_hdus = [
					(index, hdu)
					for index, hdu in enumerate(hdus)
					if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))
				]
				if table_hdus:
					hdu_index, table_hdu = table_hdus[0]
					kind = "fits_table"
					data_shape = [len(table_hdu.data)]
					axis_types = list(table_hdu.columns.names or [])
				else:
					kind = "unknown_fits"
					data_shape = None
					hdu_index = 0

		return {
			"path": str(fits_path.resolve()),
			"kind": kind,
			"hdu_index": hdu_index,
			"shape": data_shape,
			"object": object_name,
			"axis_types": axis_types,
			"extensions": sorted(extension_names),
		}