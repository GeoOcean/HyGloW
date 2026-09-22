"""Polar wave-spectrum plotting — comparing several sources side by side.

wavespectra's own ``.spec.plot()`` cannot target a pre-existing subplot axis:
it always builds its own ``subplot_kws`` for a fresh polar axis internally
(see ``wavespectra.plot.WavePlot.__call__``), which xarray rejects together
with a caller-supplied ``ax=``. So a real side-by-side comparison figure
(e.g. buoy vs. two reconstructions, laid out so the same moment lines up
across panels) can't be built by handing each source its own subplot in one
figure the usual way — each spectrum is rendered to its own image, then the
images are composed. Same technique ``00_Course_Figures.ipynb`` already uses
for its slide compositions; this is the general-purpose version of it.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import xarray as xr
from PIL import Image

DEFAULT_FIGSIZE = (5.5, 4.8)
DEFAULT_CMAP = "magma"  # sequential — spectral energy density is non-negative


def render_polar_spectrum(
    da: xr.DataArray,
    title: str,
    *,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    cmap: str = DEFAULT_CMAP,
    dpi: int = 130,
    **plot_kwargs: Any,
) -> Image.Image:
    """Render one polar wave spectrum (``da.spec.plot()``) to a raster image.

    Parameters
    ----------
    da
        A 2D (freq, dir) spectrum — or anything ``.spec.plot()`` accepts,
        e.g. a time-averaged one.
    title
        Axes title.
    figsize, cmap, dpi
        Figure size (inches), colormap, and raster resolution.
    plot_kwargs
        Extra kwargs forwarded to ``da.spec.plot()`` (e.g. ``normalised``,
        ``levels``, ``rmax``).

    Returns
    -------
    PIL.Image.Image
        Rendered figure, cropped tight, white background.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, facecolor="white")
    da.load().spec.plot(cmap=cmap, add_colorbar=True, **plot_kwargs)
    ax = plt.gca()
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def render_spectrum_bubble(
    da: xr.DataArray,
    *,
    figsize: tuple[float, float] = (1.8, 1.8),
    cmap: str = DEFAULT_CMAP,
    levels: int = 16,
    dpi: int = 130,
) -> Image.Image:
    """Render one polar spectrum as a small transparent bubble for a map.

    No title, ticks, or colorbar — meant to be stamped onto a basemap at its
    site's coordinates via ``matplotlib.offsetbox.AnnotationBbox`` /
    ``OffsetImage``, e.g. many reconstructed spectra shown at once across a
    domain rather than one large panel per spectrum (see
    :func:`render_polar_spectrum` for that case).

    Parameters
    ----------
    da
        A 2D (freq, dir) spectrum.
    figsize, cmap, levels, dpi
        Figure size (inches), colormap, contour level count, and raster
        resolution. Kept small by default since many bubbles typically share
        one map.

    Returns
    -------
    PIL.Image.Image
        Rendered bubble with a transparent background.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    da.load().spec.plot(cmap=cmap, levels=levels, normalised=True, add_colorbar=False)
    ax = plt.gca()
    ax.set_title("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.35, linewidth=0.5, color="white")

    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def stack_images_vertical(images: list[Image.Image], *, pad: int = 12) -> Image.Image:
    """Vertically stack rendered images, centred, with white padding between them."""
    width = max(im.width for im in images)
    height = sum(im.height for im in images) + pad * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for im in images:
        x = (width - im.width) // 2
        canvas.paste(im, (x, y))
        y += im.height + pad
    return canvas


def stack_images_horizontal(images: list[Image.Image], *, pad: int = 14) -> Image.Image:
    """Horizontally stack rendered images, centred, with white padding between them."""
    width = sum(im.width for im in images) + pad * (len(images) - 1)
    height = max(im.height for im in images)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for im in images:
        y = (height - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + pad
    return canvas


def plot_spectrum_comparison(
    specs: list[tuple[str, xr.DataArray]],
    *,
    orientation: str = "horizontal",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    cmap: str = DEFAULT_CMAP,
    dpi: int = 130,
    **plot_kwargs: Any,
) -> Image.Image:
    """Render several spectra for the same moment and lay them out for direct comparison.

    One panel per ``(label, spectrum)`` pair, always in the order given —
    e.g. always [("Buoy", ...), ("Bulk", ...), ("Spectral", ...)], so
    whatever else varies (one timestep, a monthly mean, a whole random
    sample), every figure has the same, directly comparable layout.

    Parameters
    ----------
    specs
        ``(panel_label, spectrum)`` pairs, one panel each, in reading order.
    orientation
        ``"horizontal"`` (default) — panels left to right, 1 row / N columns.
        ``"vertical"`` — panels top to bottom, N rows / 1 column.
    figsize, cmap, dpi, plot_kwargs
        Forwarded to :func:`render_polar_spectrum` for every panel.

    Returns
    -------
    PIL.Image.Image
        The composed comparison image — display it with IPython's
        ``display()``, or ``.save(path)`` it.
    """
    images = [
        render_polar_spectrum(da, label, figsize=figsize, cmap=cmap, dpi=dpi, **plot_kwargs)
        for label, da in specs
    ]
    if orientation == "horizontal":
        return stack_images_horizontal(images)
    if orientation == "vertical":
        return stack_images_vertical(images)
    raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {orientation!r}")


__all__ = [
    "DEFAULT_CMAP",
    "DEFAULT_FIGSIZE",
    "plot_spectrum_comparison",
    "render_polar_spectrum",
    "render_spectrum_bubble",
    "stack_images_horizontal",
    "stack_images_vertical",
]
