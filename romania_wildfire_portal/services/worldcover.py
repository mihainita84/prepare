from __future__ import annotations

import io
import math

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.mask import mask
from rasterio.windows import from_bounds
from shapely.geometry import mapping, box

# Current Terrascope public WMS endpoint and current layer identifier (2026).
WORLDCOVER_WMS = "https://titiler.terrascope.be/wms"
WORLDCOVER_WMS_LAYER = "esa-worldcover-map-10m-2021-v2_map"
WORLDCOVER_HTTP_ROOT = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

CLASS_COLORS = {
    0: (255, 255, 255, 0),
    10: (0, 100, 0, 210),
    20: (255, 187, 34, 210),
    30: (255, 255, 76, 210),
    40: (240, 150, 255, 210),
    50: (250, 0, 0, 210),
    60: (180, 180, 180, 210),
    70: (240, 240, 240, 210),
    80: (0, 100, 200, 210),
    90: (0, 150, 160, 210),
    95: (0, 207, 117, 210),
    100: (250, 230, 160, 210),
}


def _tile_code(lat0: int, lon0: int) -> str:
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}"


def _tile_url(lat0: int, lon0: int) -> str:
    code = _tile_code(lat0, lon0)
    return f"{WORLDCOVER_HTTP_ROOT}/ESA_WorldCover_10m_2021_v200_{code}_Map.tif"


def _tile_origins(bounds: tuple[float, float, float, float]):
    west, south, east, north = bounds
    lon_start = math.floor(west / 3) * 3
    lat_start = math.floor(south / 3) * 3
    lon_end = math.floor((east - 1e-10) / 3) * 3
    lat_end = math.floor((north - 1e-10) / 3) * 3
    for lat0 in range(lat_start, lat_end + 1, 3):
        for lon0 in range(lon_start, lon_end + 1, 3):
            yield lat0, lon0


def _centroid_tile_origin(aoi: gpd.GeoDataFrame) -> tuple[int, int]:
    geom = aoi.to_crs(4326).geometry.iloc[0]
    c = geom.centroid
    return math.floor(c.y / 3) * 3, math.floor(c.x / 3) * 3


def landcover_stats(aoi: gpd.GeoDataFrame, aoi_area_ha: float | None = None) -> dict:
    """Read only intersecting windows from ESA WorldCover public AWS COGs."""
    aoi = aoi.to_crs(4326)
    geom = aoi.geometry.iloc[0]
    if aoi_area_ha is None:
        aoi_area_ha = float(aoi.to_crs(3035).area.iloc[0] / 10_000.0)

    counts: dict[int, int] = {}
    tile_errors: list[str] = []
    valid_pixels = 0

    env_opts = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    }
    with rasterio.Env(**env_opts):
        for lat0, lon0 in _tile_origins(tuple(aoi.total_bounds)):
            tile_geom = box(lon0, lat0, lon0 + 3, lat0 + 3)
            inter = geom.intersection(tile_geom)
            if inter.is_empty:
                continue
            url = _tile_url(lat0, lon0)
            try:
                with rasterio.open(url) as src:
                    data, _ = mask(src, [mapping(inter)], crop=True, all_touched=True, filled=False)
                    arr = data[0]
                    vals = arr.compressed()
                    if src.nodata is not None:
                        vals = vals[vals != src.nodata]
                    vals = vals[vals != 0]
                    if vals.size == 0:
                        continue
                    u, c = np.unique(vals.astype(int), return_counts=True)
                    for klass, n in zip(u, c):
                        counts[int(klass)] = counts.get(int(klass), 0) + int(n)
                    valid_pixels += int(vals.size)
            except Exception as exc:
                tile_errors.append(f"{_tile_code(lat0, lon0)}: {exc}")

    if valid_pixels == 0:
        detail = "; ".join(tile_errors[:3])
        raise RuntimeError("ESA WorldCover online COGs returned no valid pixels." + (f" {detail}" if detail else ""))

    shares = {k: 100.0 * v / valid_pixels for k, v in counts.items()}
    forest_pct = shares.get(10, 0.0)
    class_breakdown = {
        CLASSES.get(k, f"Class {k}"): {
            "class_code": k,
            "percent": round(shares[k], 3),
            "area_ha_est": round(aoi_area_ha * shares[k] / 100.0, 3),
        }
        for k in sorted(shares)
    }
    return {
        "forest_area_ha": float(aoi_area_ha * forest_pct / 100.0),
        "forest_fraction_pct": float(forest_pct),
        "worldcover_valid_pixels": valid_pixels,
        "worldcover_classes": class_breakdown,
        "worldcover_year": 2021,
        "worldcover_resolution_m": 10,
        "worldcover_source": "ESA WorldCover 2021 v200 public AWS COGs",
        "worldcover_tile_warnings": tile_errors[:5],
    }


def _context_bounds(aoi: gpd.GeoDataFrame, pad_frac: float = 0.20, min_pad_deg: float = 0.01) -> tuple[float, float, float, float]:
    geom = aoi.to_crs(4326).geometry.iloc[0]
    west, south, east, north = geom.bounds
    dx = max(east - west, min_pad_deg)
    dy = max(north - south, min_pad_deg)
    pad_x = dx * pad_frac + min_pad_deg
    pad_y = dy * pad_frac + min_pad_deg
    return west - pad_x, south - pad_y, east + pad_x, north + pad_y


def _rgba_from_classes(data: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    for klass, color in CLASS_COLORS.items():
        mask = data == klass
        if np.any(mask):
            rgba[mask] = color
    return rgba


def _draw_polygon_outline(img: Image.Image, aoi_bounds: tuple[float, float, float, float], ctx_bounds: tuple[float, float, float, float]) -> Image.Image:
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


def worldcover_thumbnail(aoi: gpd.GeoDataFrame, max_size: int = 420) -> bytes:
    """Generate a simple PNG thumbnail around the AOI from the tile containing its centroid.

    This is for user-friendly previews. AOI statistics continue to use all intersecting tiles.
    """
    aoi = aoi.to_crs(4326)
    ctx_bounds = _context_bounds(aoi)
    aoi_bounds = tuple(aoi.total_bounds)
    lat0, lon0 = _centroid_tile_origin(aoi)
    url = _tile_url(lat0, lon0)

    env_opts = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    }
    with rasterio.Env(**env_opts):
        with rasterio.open(url) as src:
            win = from_bounds(*ctx_bounds, transform=src.transform)
            win = win.round_offsets().round_lengths()
            data = src.read(1, window=win, out_shape=(max_size, max_size), resampling=rasterio.enums.Resampling.nearest)
    rgba = _rgba_from_classes(data)
    img = Image.fromarray(rgba, mode="RGBA")
    img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    img = _draw_polygon_outline(img, aoi_bounds, ctx_bounds)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
