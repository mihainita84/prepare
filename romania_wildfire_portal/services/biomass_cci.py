from __future__ import annotations

import io
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
import requests
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.windows import from_bounds
from shapely.geometry import box, mapping

CCI_VERSION = "7.0"
CCI_YEAR = 2024
CCI_RESOLUTION_M = 100
CEDA_ROOT = (
    "https://dap.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/"
    f"v{CCI_VERSION}/geotiff/{CCI_YEAR}"
)

CACHE_DIR = Path(os.getenv("WILDFIRE_CACHE_DIR", Path(tempfile.gettempdir()) / "romania_wildfire_portal_cache")) / "biomass_cci"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _lon_tile_origin(value: float) -> int:
    return math.floor(value / 10.0) * 10


def _lat_tile_north(value: float) -> int:
    return math.ceil(value / 10.0) * 10


def _tile_code(lat0: int, lon0: int) -> str:
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}"


def _tile_filename(lat0: int, lon0: int, uncertainty: bool = False) -> str:
    code = _tile_code(lat0, lon0)
    variable = "AGB_SD" if uncertainty else "AGB"
    return f"{code}_ESACCI-BIOMASS-L4-{variable}-MERGED-100m-{CCI_YEAR}-fv{CCI_VERSION}.tif"


def _tile_url(lat0: int, lon0: int, uncertainty: bool = False) -> str:
    return f"{CEDA_ROOT}/{_tile_filename(lat0, lon0, uncertainty)}"


def _tile_origins(bounds: tuple[float, float, float, float]) -> Iterable[tuple[int, int]]:
    west, south, east, north = bounds
    lon_start = _lon_tile_origin(west)
    lon_end = _lon_tile_origin(east - 1e-10)
    north_start = _lat_tile_north(south + 1e-10)
    north_end = _lat_tile_north(north - 1e-10)
    for north_edge in range(north_start, north_end + 1, 10):
        for west_edge in range(lon_start, lon_end + 1, 10):
            yield north_edge, west_edge


def _centroid_tile_origin(aoi: gpd.GeoDataFrame) -> tuple[int, int]:
    geom = aoi.to_crs(4326).geometry.iloc[0]
    c = geom.centroid
    return _lat_tile_north(c.y), _lon_tile_origin(c.x)


def _download_tile(lat0: int, lon0: int, uncertainty: bool = False) -> Path:
    filename = _tile_filename(lat0, lon0, uncertainty)
    target = CACHE_DIR / filename
    if target.exists() and target.stat().st_size > 100_000:
        return target

    url = _tile_url(lat0, lon0, uncertainty)
    partial = target.with_suffix(target.suffix + ".part")
    headers = {"User-Agent": "PREPARE-Romania-Wildfire-Portal/4.0"}
    try:
        with requests.get(url, stream=True, timeout=(20, 180), headers=headers, allow_redirects=True) as r:
            r.raise_for_status()
            ctype = (r.headers.get("content-type") or "").lower()
            if "html" in ctype:
                raise RuntimeError(f"CEDA returned HTML instead of GeoTIFF for {filename}")
            with open(partial, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        if partial.stat().st_size < 100_000:
            raise RuntimeError(f"Downloaded CCI Biomass tile is unexpectedly small: {filename}")
        os.replace(partial, target)
    finally:
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass
    return target


def _valid_values(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    vals = np.asarray(arr).astype("float64", copy=False).ravel()
    vals = vals[np.isfinite(vals)]
    if nodata is not None:
        vals = vals[vals != nodata]
    return vals[(vals >= 0) & (vals < 5000)]


def biomass_stats(aoi: gpd.GeoDataFrame, aoi_area_ha: float | None = None) -> dict:
    aoi = aoi.to_crs(4326)
    geom = aoi.geometry.iloc[0]
    if aoi_area_ha is None:
        aoi_area_ha = float(aoi.to_crs(3035).area.iloc[0] / 10_000.0)

    agb_parts: list[np.ndarray] = []
    sd_parts: list[np.ndarray] = []
    warnings: list[str] = []

    for lat0, lon0 in _tile_origins(tuple(aoi.total_bounds)):
        tile_geom = box(lon0, lat0 - 10, lon0 + 10, lat0)
        inter = geom.intersection(tile_geom)
        if inter.is_empty:
            continue
        try:
            agb_path = _download_tile(lat0, lon0, uncertainty=False)
            with rasterio.open(agb_path) as src:
                data, _ = mask(src, [mapping(inter)], crop=True, all_touched=True, filled=False)
                vals = data[0].compressed() if np.ma.isMaskedArray(data[0]) else data[0].ravel()
                vals = _valid_values(vals, src.nodata)
                if vals.size:
                    agb_parts.append(vals)
        except Exception as exc:
            warnings.append(f"AGB {_tile_code(lat0, lon0)}: {exc}")

        try:
            sd_path = _download_tile(lat0, lon0, uncertainty=True)
            with rasterio.open(sd_path) as src:
                data, _ = mask(src, [mapping(inter)], crop=True, all_touched=True, filled=False)
                vals = data[0].compressed() if np.ma.isMaskedArray(data[0]) else data[0].ravel()
                vals = _valid_values(vals, src.nodata)
                if vals.size:
                    sd_parts.append(vals)
        except Exception as exc:
            warnings.append(f"AGB_SD {_tile_code(lat0, lon0)}: {exc}")

    if not agb_parts:
        detail = "; ".join(warnings[:3])
        raise RuntimeError("ESA CCI Biomass returned no valid AGB pixels." + (f" {detail}" if detail else ""))

    agb = np.concatenate(agb_parts)
    sd = np.concatenate(sd_parts) if sd_parts else np.array([], dtype="float64")
    positive = agb[agb > 0]
    mean_agb = float(np.mean(agb))

    return {
        "agb_mean_mg_ha": mean_agb,
        "agb_median_mg_ha": float(np.median(agb)),
        "agb_p90_mg_ha": float(np.percentile(agb, 90)),
        "agb_mean_positive_mg_ha": float(np.mean(positive)) if positive.size else 0.0,
        "agb_standard_deviation_mean_mg_ha": float(np.mean(sd)) if sd.size else None,
        "agb_pixels": int(agb.size),
        "agb_total_aoi_mg_est": float(mean_agb * aoi_area_ha),
        "source": f"ESA CCI Biomass v{CCI_VERSION} ({CCI_YEAR}) via CEDA public GeoTIFFs",
        "year": CCI_YEAR,
        "resolution_m": CCI_RESOLUTION_M,
        "biomass_tile_warnings": warnings[:5],
    }


def _rgba_from_agb(data: np.ndarray, nodata: float | int | None, vmax: float = 300.0) -> np.ndarray:
    arr = data.astype("float32", copy=False)
    valid = np.isfinite(arr) & (arr >= 0) & (arr < 5000)
    if nodata is not None:
        valid &= arr != nodata
    visible = valid & (arr > 0)
    scaled = np.clip(arr / vmax, 0.0, 1.0)
    stops = np.array([0.0, 0.25, 0.55, 1.0], dtype="float32")
    colors = np.array([
        [255, 247, 188],
        [173, 221, 142],
        [49, 163, 84],
        [0, 90, 50],
    ], dtype="float32")
    rgb = np.zeros((*arr.shape, 3), dtype="uint8")
    for band in range(3):
        rgb[..., band] = np.interp(scaled, stops, colors[:, band]).astype("uint8")
    alpha = np.where(visible, 190, 0).astype("uint8")
    return np.dstack([rgb, alpha])


def biomass_overlay_png(
    bounds: tuple[float, float, float, float] = (20.0, 43.4, 30.0, 48.5),
    max_width: int = 950,
) -> tuple[bytes, list[list[float]]]:
    west, south, east, north = bounds
    lat0, lon0 = _lat_tile_north(north - 1e-10), _lon_tile_origin(west)
    if _lat_tile_north(south + 1e-10) != lat0 or _lon_tile_origin(east - 1e-10) != lon0:
        raise ValueError("Display overlay bounds cross an ESA CCI 10-degree tile boundary.")

    path = _download_tile(lat0, lon0, uncertainty=False)
    with rasterio.open(path) as src:
        win = from_bounds(west, south, east, north, src.transform)
        win = win.round_offsets().round_lengths()
        aspect = max(0.25, min(4.0, (north - south) / max(1e-9, east - west)))
        width = max_width
        height = max(300, int(width * aspect))
        data = src.read(1, window=win, out_shape=(height, width), resampling=Resampling.average)
        rgba = _rgba_from_agb(data, src.nodata)

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    leaflet_bounds = [[south, west], [north, east]]
    return buf.getvalue(), leaflet_bounds


def _context_bounds(aoi: gpd.GeoDataFrame, pad_frac: float = 0.20, min_pad_deg: float = 0.01) -> tuple[float, float, float, float]:
    geom = aoi.to_crs(4326).geometry.iloc[0]
    west, south, east, north = geom.bounds
    dx = max(east - west, min_pad_deg)
    dy = max(north - south, min_pad_deg)
    pad_x = dx * pad_frac + min_pad_deg
    pad_y = dy * pad_frac + min_pad_deg
    return west - pad_x, south - pad_y, east + pad_x, north + pad_y


def _draw_aoi_outline(img: Image.Image, aoi_bounds: tuple[float, float, float, float], ctx_bounds: tuple[float, float, float, float]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, s, e, n = ctx_bounds
    aw, as_, ae, an = aoi_bounds
    def px(x):
        return int((x - w) / max(1e-9, e - w) * (img.width - 1))
    def py(y):
        return int((n - y) / max(1e-9, n - s) * (img.height - 1))
    pts = [(px(aw), py(as_)), (px(ae), py(as_)), (px(ae), py(an)), (px(aw), py(an)), (px(aw), py(as_))]
    draw.line(pts, fill=(0, 0, 0, 255), width=3)
    return img


def biomass_thumbnail(aoi: gpd.GeoDataFrame, max_size: int = 420) -> bytes:
    aoi = aoi.to_crs(4326)
    ctx_bounds = _context_bounds(aoi)
    aoi_bounds = tuple(aoi.total_bounds)
    lat0, lon0 = _centroid_tile_origin(aoi)
    path = _download_tile(lat0, lon0, uncertainty=False)
    with rasterio.open(path) as src:
        win = from_bounds(*ctx_bounds, transform=src.transform)
        win = win.round_offsets().round_lengths()
        data = src.read(1, window=win, out_shape=(max_size, max_size), resampling=Resampling.average)
        rgba = _rgba_from_agb(data, src.nodata)
    img = Image.fromarray(rgba, mode="RGBA")
    img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    img = _draw_aoi_outline(img, aoi_bounds, ctx_bounds)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
