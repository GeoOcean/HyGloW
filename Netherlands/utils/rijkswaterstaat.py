"""Rijkswaterstaat (Waterinfo/DDL) raw wave-buoy NetCDF -> HyGloW processed schema."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from utils.config import resolve

VALID_QUALITY_CODE = "00"
"""``Kwaliteitswaardecode`` kept as valid.

Nearly all rows are ``'00'`` (validated) or ``'99'``/``''`` (missing, already
NaN in the raw value). A small remainder (~1-2%, codes ``1080``-``1087`` and
``25``) is of unverified provenance in this export and is dropped along with
the missing rows rather than guessed at.
"""


def _valid(raw: xr.Dataset, var: str) -> xr.DataArray:
    """*var* values where its quality code is :data:`VALID_QUALITY_CODE`, else NaN."""
    code = raw[f"{var}__WaarnemingMetadata_Kwaliteitswaardecode"]
    return raw[var].where(code == VALID_QUALITY_CODE)


def process_raw_buoy(path: Path | str) -> xr.Dataset:
    """Convert one raw RWS wave-buoy NetCDF to the HyGloW ``hs``/``dir`` schema.

    Rijkswaterstaat's Waterinfo/DDL export reports ``Hm0`` in cm (not m) and
    fills invalid samples with literal sentinels (``Th0`` up to 99999,
    ``Hm0`` itself occasionally 99999 cm) even inside otherwise-"valid"
    (``Kwaliteitswaardecode == '00'``) rows — the same class of bug fixed in
    notebook 09's dynamic-output comparison (see ``PHYSICAL_BOUNDS`` in
    ``utils/validation.py``), so it is masked here at the source
    instead of trusting the quality code alone.

    Only ``hs`` (from ``Hm0``) and ``dir`` (from ``Th0``) are produced.
    ``Tm-10`` (spectral mean period) is intentionally dropped rather than
    relabeled ``tp``: it is a different statistic from the peak period the
    metamodel predicts, and mixing them would bias any ``tp`` comparison.

    Parameters
    ----------
    path
        Raw NetCDF path, as downloaded from Waterinfo/DDL.

    Returns
    -------
    xarray.Dataset
        ``(time,)`` with ``hs`` (m, physically masked to ``[0, 20)``) and
        ``dir`` (deg, masked to ``[0, 360)``).
    """
    raw = xr.open_dataset(resolve(path))

    hs = _valid(raw, "Hm0") / 100.0  # cm -> m
    hs = hs.where((hs >= 0.0) & (hs < 20.0))

    direction = _valid(raw, "Th0")
    direction = direction.where((direction >= 0.0) & (direction < 360.0))

    out = xr.Dataset({"hs": hs, "dir": direction})
    out["hs"].attrs = {
        "standard_name": "sea_surface_wave_significant_height",
        "units": "m",
    }
    out["dir"].attrs = {
        "standard_name": "sea_surface_wave_from_direction",
        "units": "degree",
    }
    out.attrs["source"] = "Rijkswaterstaat Waterinfo/DDL"
    out.attrs["station_id"] = raw.attrs.get("station_id", "")
    return out


__all__ = ["VALID_QUALITY_CODE", "process_raw_buoy"]
