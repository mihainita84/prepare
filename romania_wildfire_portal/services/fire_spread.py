from __future__ import annotations

import heapq
import io
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiLineString, Point, box, mapping
from shapely.ops import transform as shp_transform, unary_union
from shapely.prepared import prep

from services.biomass_cci import _download_tile as _cci_download_tile
from services.biomass_cci import _lat_tile_north as _cci_lat_tile_north
from services.biomass_cci import _lon_tile_origin as _cci_lon_tile_origin
from services.worldcover import WORLDCOVER_HTTP_ROOT

# -----------------------------------------------------------------------------
# EXPERIMENTAL SCREENING MODEL
# -----------------------------------------------------------------------------
# This module intentionally does not claim to be a calibrated operational
# implementation of Rothermel/BehavePlus. It is a directional travel-time model
# inspired by the same main controls: fuel, moisture, wind and slope. Its purpose
# is scenario comparison inside the PREPARE research portal.

# Baseline surface spread rates (m/min) for scenario screening under moderate,
# dry-ish conditions. These are relative model coefficients, not prescriptions.
BASE_ROS_M_MIN = {
    # Internal screening coefficients. v13 deliberately reduces the previous
    # background rates so that landscape fuel continuity, biomass, moisture,
    # slope and wind have visible control over the resulting footprint.
    10: 0.90,  # tree cover
    20: 2.40,  # shrubland
    30: 4.80,  # grassland
    40: 1.50,  # cropland
    50: 0.0,   # built-up = non-wildland fuel
    60: 0.12,  # bare / sparse
    70: 0.0,   # snow / ice
    80: 0.0,   # water
    90: 0.20,  # herbaceous wetland
    95: 1.00,  # mangroves (not expected in Romania)
    100: 0.45, # moss / lichen
    0: 0.0,    # unknown is not allowed to carry fire in v13
}

FUEL_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow / ice",
    80: "Water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss / lichen",
    0: "Unknown",
}

# Plausibility caps for this screening model. They prevent a low-resolution
# landscape cell from attaining extreme rates simply because wind and slope
# multipliers stack. These are not operational prescriptions.
MAX_ROS_M_MIN_BY_CLASS = {
    10: 8.0,
    20: 14.0,
    30: 24.0,
    40: 10.0,
    60: 1.5,
    90: 2.0,
    95: 6.0,
    100: 3.0,
    0: 0.0,
}


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _worldcover_tile_url(lat0: int, lon0: int) -> str:
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    code = f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}"
    return f"{WORLDCOVER_HTTP_ROOT}/ESA_WorldCover_10m_2021_v200_{code}_Map.tif"


def _worldcover_tile_origin(lat: float, lon: float) -> tuple[int, int]:
    return math.floor(lat / 3.0) * 3, math.floor(lon / 3.0) * 3


def _dem_tile_url(lat0: int, lon0: int) -> str:
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    northing = f"{ns}{abs(lat0):02d}_00"
    easting = f"{ew}{abs(lon0):03d}_00"
    name = f"Copernicus_DSM_COG_10_{northing}_{easting}_DEM"
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def _dem_tile_origin(lat: float, lon: float) -> tuple[int, int]:
    return math.floor(lat), math.floor(lon)


def _sample_grouped_remote(
    lons: np.ndarray,
    lats: np.ndarray,
    valid_flat: np.ndarray,
    tile_fn,
    url_fn,
    fallback: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sample remote COGs efficiently and retain a validity mask.

    v13 distinguishes a genuine low/zero value from a failed remote lookup.
    This is important because missing WorldCover/DEM data must not silently
    become generic burnable landscape.
    """
    out = np.full(lons.shape, fallback, dtype="float32")
    sample_valid = np.zeros(lons.shape, dtype=bool)
    warnings: list[str] = []
    groups: dict[tuple[int, int], list[int]] = {}
    idxs = np.flatnonzero(valid_flat)
    for idx in idxs:
        key = tile_fn(float(lats[idx]), float(lons[idx]))
        groups.setdefault(key, []).append(int(idx))

    env_opts = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    }
    with rasterio.Env(**env_opts):
        for key, group_idxs in groups.items():
            try:
                url = url_fn(*key)
                with rasterio.open(url) as src:
                    xs = [float(lons[i]) for i in group_idxs]
                    ys = [float(lats[i]) for i in group_idxs]
                    rows, cols = rowcol(src.transform, xs, ys)
                    rows = np.asarray(rows, dtype=int)
                    cols = np.asarray(cols, dtype=int)
                    inside_src = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
                    if not inside_src.any():
                        continue
                    rmin, rmax = int(rows[inside_src].min()), int(rows[inside_src].max())
                    cmin, cmax = int(cols[inside_src].min()), int(cols[inside_src].max())
                    win = Window(cmin, rmin, cmax - cmin + 1, rmax - rmin + 1)
                    data = src.read(1, window=win, masked=True)
                    vals = np.full(len(group_idxs), fallback, dtype="float32")
                    valid_local = np.zeros(len(group_idxs), dtype=bool)
                    local_r = rows - rmin
                    local_c = cols - cmin
                    ok = inside_src & (local_r >= 0) & (local_r < data.shape[0]) & (local_c >= 0) & (local_c < data.shape[1])
                    for j in np.flatnonzero(ok):
                        v = data[int(local_r[j]), int(local_c[j])]
                        if np.ma.is_masked(v):
                            continue
                        fv = float(v)
                        if np.isfinite(fv):
                            vals[j] = fv
                            valid_local[j] = True
                    out[group_idxs] = vals
                    sample_valid[np.asarray(group_idxs, dtype=int)] = valid_local
            except Exception as exc:
                warnings.append(f"{key}: {exc}")
    return out, sample_valid, warnings


def _sample_biomass(
    lons: np.ndarray,
    lats: np.ndarray,
    valid_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sample ESA CCI AGB and retain which cells have genuine source values."""
    out = np.zeros(lons.shape, dtype="float32")
    sample_valid = np.zeros(lons.shape, dtype=bool)
    warnings: list[str] = []
    groups: dict[tuple[int, int], list[int]] = {}
    for idx in np.flatnonzero(valid_flat):
        key = (_cci_lat_tile_north(float(lats[idx])), _cci_lon_tile_origin(float(lons[idx])))
        groups.setdefault(key, []).append(int(idx))

    for (north_edge, west_edge), group_idxs in groups.items():
        try:
            path = _cci_download_tile(north_edge, west_edge, uncertainty=False)
            with rasterio.open(path) as src:
                coords = [(float(lons[i]), float(lats[i])) for i in group_idxs]
                vals = []
                valid_local = []
                for v in src.sample(coords, indexes=1, masked=True):
                    if np.ma.is_masked(v[0]):
                        vals.append(0.0)
                        valid_local.append(False)
                        continue
                    x = float(v[0])
                    ok = bool(np.isfinite(x) and 0 <= x <= 5000 and (src.nodata is None or x != src.nodata))
                    vals.append(x if ok else 0.0)
                    valid_local.append(ok)
                out[group_idxs] = np.asarray(vals, dtype="float32")
                sample_valid[np.asarray(group_idxs, dtype=int)] = np.asarray(valid_local, dtype=bool)
        except Exception as exc:
            warnings.append(f"N{north_edge} E{west_edge}: {exc}")
    return out, sample_valid, warnings


def _nearest_fill(arr: np.ndarray, valid_mask: np.ndarray, default: float = 0.0) -> np.ndarray:
    x = np.asarray(arr, dtype="float32").copy()
    bad = ~np.isfinite(x) | (~valid_mask)
    good_vals = x[np.isfinite(x) & valid_mask]
    fill = float(np.nanmedian(good_vals)) if good_vals.size else float(default)
    x[bad] = fill
    return x


def build_environment(aoi: gpd.GeoDataFrame, cell_size_m: int = 100) -> dict:
    """Build a compact 100-m landscape grid for spread-scenario screening."""
    aoi4326 = aoi.to_crs(4326)
    geom4326 = aoi4326.geometry.iloc[0]
    c = geom4326.centroid
    epsg = _utm_epsg(c.x, c.y)
    aoi_utm = aoi4326.to_crs(epsg)
    geom_utm = aoi_utm.geometry.iloc[0]
    minx, miny, maxx, maxy = geom_utm.bounds

    # Pad to whole cells and cap grid dimensions defensively.
    minx = math.floor(minx / cell_size_m) * cell_size_m
    miny = math.floor(miny / cell_size_m) * cell_size_m
    maxx = math.ceil(maxx / cell_size_m) * cell_size_m
    maxy = math.ceil(maxy / cell_size_m) * cell_size_m
    ncols = max(1, int(round((maxx - minx) / cell_size_m)))
    nrows = max(1, int(round((maxy - miny) / cell_size_m)))
    if ncols * nrows > 25_000:
        raise RuntimeError("Spread grid exceeds 25,000 cells; use a smaller AOI or a coarser grid.")

    xs = minx + (np.arange(ncols) + 0.5) * cell_size_m
    ys = maxy - (np.arange(nrows) + 0.5) * cell_size_m
    xx, yy = np.meshgrid(xs, ys)

    pg = prep(geom_utm)
    inside = np.zeros((nrows, ncols), dtype=bool)
    for r in range(nrows):
        for col in range(ncols):
            inside[r, col] = pg.contains(Point(float(xx[r, col]), float(yy[r, col])))

    transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon_flat, lat_flat = transformer.transform(xx.ravel(), yy.ravel())
    lon_flat = np.asarray(lon_flat)
    lat_flat = np.asarray(lat_flat)
    valid_flat = inside.ravel()

    # Spatial fuels from ESA WorldCover.
    wc_flat, wc_valid_flat, wc_warn = _sample_grouped_remote(
        lon_flat,
        lat_flat,
        valid_flat,
        _worldcover_tile_origin,
        _worldcover_tile_url,
        fallback=0.0,
    )
    worldcover = wc_flat.reshape(nrows, ncols).astype("int16")
    wc_valid = wc_valid_flat.reshape(nrows, ncols)
    wc_coverage_pct = 100.0 * float((wc_valid & inside).sum()) / max(int(inside.sum()), 1)
    if wc_coverage_pct < 90.0:
        raise RuntimeError(
            f"WorldCover coverage is only {wc_coverage_pct:.1f}% of the spread grid. "
            "The simulator stops rather than treating missing land-cover data as burnable fuel."
        )

    # Biomass context from ESA CCI.
    biomass_flat, biomass_valid_flat, bio_warn = _sample_biomass(lon_flat, lat_flat, valid_flat)
    biomass = biomass_flat.reshape(nrows, ncols)
    biomass_valid = biomass_valid_flat.reshape(nrows, ncols)

    # Terrain from Copernicus DEM GLO-30 COGs.
    dem_flat, dem_valid_flat, dem_warn = _sample_grouped_remote(
        lon_flat,
        lat_flat,
        valid_flat,
        _dem_tile_origin,
        _dem_tile_url,
        fallback=np.nan,
    )
    dem_raw = dem_flat.reshape(nrows, ncols)
    dem_valid = dem_valid_flat.reshape(nrows, ncols)
    dem = _nearest_fill(dem_raw, inside, default=0.0)

    # Elevation gradients: northing increases upward, array rows increase southward.
    dz_dy, dz_dx = np.gradient(dem, -cell_size_m, cell_size_m)
    slope_tan = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    slope_deg = np.degrees(np.arctan(slope_tan))

    # Lightweight hillshade retained for the exported animation so the terrain
    # used by the model is also visible to the user.
    slope_rad = np.arctan(slope_tan)
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    azimuth = math.radians(315.0)
    altitude = math.radians(45.0)
    hill = (
        math.sin(altitude) * np.cos(slope_rad)
        + math.cos(altitude) * np.sin(slope_rad) * np.cos(azimuth - aspect_rad)
    )
    hillshade = np.clip(0.72 + 0.38 * ((hill + 1.0) / 2.0), 0.68, 1.10).astype("float32")

    base_ros = np.zeros_like(dem, dtype="float32")
    for klass, rate in BASE_ROS_M_MIN.items():
        base_ros[worldcover == klass] = float(rate)
    # Missing/unknown land cover no longer receives a generic positive spread rate.
    base_ros[(inside) & (~wc_valid)] = 0.0
    base_ros[~inside] = 0.0

    # ESA CCI AGB is woody above-ground biomass, not surface-fuel load. v13 uses
    # it as a *continuity/structure limiter* for woody WorldCover classes:
    # low AGB slows woody-fuel spread substantially; medium/high AGB saturates
    # near 1.0 instead of unrealistically accelerating ROS without bound.
    biomass_factor = np.ones_like(dem, dtype="float32")
    tree = inside & (worldcover == 10)
    shrub = inside & (worldcover == 20)

    tree_valid = tree & biomass_valid
    shrub_valid = shrub & biomass_valid
    biomass_factor[tree_valid] = (
        0.22 + 0.78 * (1.0 - np.exp(-np.clip(biomass[tree_valid], 0, 400) / 85.0))
    ).astype("float32")
    biomass_factor[shrub_valid] = (
        0.55 + 0.45 * (1.0 - np.exp(-np.clip(biomass[shrub_valid], 0, 250) / 55.0))
    ).astype("float32")

    # If CCI is unavailable at a cell, do not pretend AGB=0; leave factor neutral.
    biomass_factor[(tree | shrub) & (~biomass_valid)] = 1.0
    biomass_factor[~inside] = 1.0

    # A compact fuel-potential layer is retained for diagnostics/video rendering.
    fuel_potential = base_ros * biomass_factor

    # WGS84 bounds of projected grid corners for Leaflet ImageOverlay.
    corners_x = [minx, maxx, maxx, minx]
    corners_y = [miny, miny, maxy, maxy]
    corner_lon, corner_lat = transformer.transform(corners_x, corners_y)
    west, east = min(corner_lon), max(corner_lon)
    south, north = min(corner_lat), max(corner_lat)

    return {
        "epsg": epsg,
        "cell_size_m": int(cell_size_m),
        "nrows": nrows,
        "ncols": ncols,
        "minx": float(minx),
        "maxy": float(maxy),
        "inside": inside,
        "worldcover": worldcover,
        "biomass": biomass.astype("float32"),
        "biomass_valid": biomass_valid,
        "dem": dem.astype("float32"),
        "dem_valid": dem_valid,
        "slope_deg": slope_deg.astype("float32"),
        "hillshade": hillshade,
        "base_ros": base_ros.astype("float32"),
        "biomass_factor": biomass_factor.astype("float32"),
        "fuel_potential": fuel_potential.astype("float32"),
        "data_quality": {
            "worldcover_coverage_pct": float(wc_coverage_pct),
            "biomass_coverage_pct": 100.0 * float((biomass_valid & inside).sum()) / max(int(inside.sum()), 1),
            "dem_coverage_pct": 100.0 * float((dem_valid & inside).sum()) / max(int(inside.sum()), 1),
        },
        "leaflet_bounds": [[float(south), float(west)], [float(north), float(east)]],
        "warnings": {
            "worldcover": wc_warn[:3],
            "biomass": bio_warn[:3],
            "dem": dem_warn[:3],
        },
    }


def estimate_fine_fuel_moisture_proxy(weather: dict) -> float:
    """Simple scenario default from RH, temperature and precipitation.

    This is explicitly a proxy, not a measured FFMC or fuel-stick moisture value.
    """
    rh = float(weather.get("relative_humidity_2m") or 55.0)
    temp = float(weather.get("temperature_2m") or 20.0)
    rain = float(weather.get("precipitation") or 0.0)
    proxy = 5.0 + 0.22 * rh - 0.10 * max(temp - 15.0, 0.0) + 3.0 * min(rain, 3.0)
    return float(np.clip(proxy, 5.0, 40.0))


def _transform_geometry_to_epsg(geom, epsg: int):
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    return shp_transform(transformer.transform, geom)


def control_barrier_geometry(env: dict, lines_geojson: list[dict] | None, width_m: float):
    """Return the exact buffered control-line geometry in the model CRS.

    The barrier is vector geometry, not only a raster mask. This is important
    because the spread graph includes 16 directional links and some links are
    longer than one cell. Testing the actual source-destination segment against
    the buffered geometry prevents the front from numerically jumping across a
    wide control strip.
    """
    if not lines_geojson:
        return None

    line_geoms = []
    for item in lines_geojson:
        geom = item.get("geometry", item) if isinstance(item, dict) else None
        gtype = geom.get("type") if isinstance(geom, dict) else None
        coords = geom.get("coordinates") if isinstance(geom, dict) else None
        if gtype == "LineString" and coords:
            line_geoms.append(LineString(coords))
        elif gtype == "MultiLineString" and coords:
            line_geoms.append(MultiLineString(coords))
    if not line_geoms:
        return None

    transformed = [_transform_geometry_to_epsg(g, env["epsg"]) for g in line_geoms]
    merged = unary_union(transformed)
    # Flat caps make the nominal line length explicit; the line can still be
    # bypassed around an end if it does not span the relevant part of the AOI.
    return merged.buffer(max(float(width_m), 1.0) / 2.0, cap_style=2, join_style=2)


def control_mask_from_lines(env: dict, lines_geojson: list[dict] | None, width_m: float) -> np.ndarray:
    """Raster display/diagnostic mask of the exact vector control strip.

    A cell is marked when its footprint intersects the barrier, rather than
    only when the cell centre falls inside it. This makes 100–200 m control
    strips visually continuous on a 50–100 m model grid.
    """
    mask = np.zeros((env["nrows"], env["ncols"]), dtype=bool)
    barrier = control_barrier_geometry(env, lines_geojson, width_m)
    if barrier is None or barrier.is_empty:
        return mask

    pbar = prep(barrier)
    cell = float(env["cell_size_m"])
    for r in range(env["nrows"]):
        y1 = env["maxy"] - r * cell
        y0 = y1 - cell
        for c in range(env["ncols"]):
            if not env["inside"][r, c]:
                continue
            x0 = env["minx"] + c * cell
            x1 = x0 + cell
            mask[r, c] = pbar.intersects(box(x0, y0, x1, y1))
    return mask


def _cell_center_xy(env: dict, r: int, c: int) -> tuple[float, float]:
    x = env["minx"] + (c + 0.5) * env["cell_size_m"]
    y = env["maxy"] - (r + 0.5) * env["cell_size_m"]
    return float(x), float(y)


def _barrier_intersection_length(env: dict, barrier_geom, r: int, c: int, nr: int, nc: int) -> float:
    """Length in metres of a propagation edge that lies inside the barrier."""
    if barrier_geom is None or barrier_geom.is_empty:
        return 0.0
    x0, y0 = _cell_center_xy(env, r, c)
    x1, y1 = _cell_center_xy(env, nr, nc)
    segment = LineString([(x0, y0), (x1, y1)])
    if not segment.intersects(barrier_geom):
        return 0.0
    inter = segment.intersection(barrier_geom)
    return float(inter.length) if not inter.is_empty else 0.0


def _ignition_cell(env: dict, lon: float, lat: float, burnable: np.ndarray) -> tuple[int, int]:
    transformer = Transformer.from_crs(4326, env["epsg"], always_xy=True)
    x, y = transformer.transform(lon, lat)
    c = int((x - env["minx"]) // env["cell_size_m"])
    r = int((env["maxy"] - y) // env["cell_size_m"])
    r = int(np.clip(r, 0, env["nrows"] - 1))
    c = int(np.clip(c, 0, env["ncols"] - 1))
    if burnable[r, c]:
        return r, c

    candidates = np.argwhere(burnable)
    if not len(candidates):
        raise RuntimeError("No burnable cells are available inside the AOI.")
    d2 = (candidates[:, 0] - r) ** 2 + (candidates[:, 1] - c) ** 2
    rr, cc = candidates[int(np.argmin(d2))]
    return int(rr), int(cc)


def _bearing_from_delta(dc: int, dr: int) -> float:
    # Array row positive = south. Convert to east/north vector and return compass bearing.
    east = float(dc)
    north = float(-dr)
    return (math.degrees(math.atan2(east, north)) + 360.0) % 360.0


def _angle_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def _scenario_moisture_factor(fine_fuel_moisture_pct: float) -> float:
    # High fine-fuel moisture strongly damps spread; low moisture accelerates it.
    return float(np.clip(1.65 - fine_fuel_moisture_pct / 30.0, 0.25, 1.5))


def _edge_traversed_cells(r: int, c: int, nr: int, nc: int) -> list[tuple[int, int]]:
    """Cells intersected by a centre-to-centre propagation edge.

    The 16-direction neighbourhood contains (2,1)/(1,2) links. Without this
    check those links can leap over a one-cell water, built-up or no-data gap.
    """
    dr = nr - r
    dc = nc - c
    length_cells = math.hypot(float(dr), float(dc))
    samples = max(8, int(math.ceil(length_cells * 10)))
    out: list[tuple[int, int]] = []
    seen = set()
    for i in range(samples + 1):
        t = i / samples
        yy = (r + 0.5) + t * dr
        xx = (c + 0.5) + t * dc
        rr = int(math.floor(yy))
        cc = int(math.floor(xx))
        key = (rr, cc)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _edge_crosses_nonfuel(burnable: np.ndarray, r: int, c: int, nr: int, nc: int) -> bool:
    for rr, cc in _edge_traversed_cells(r, c, nr, nc):
        if rr < 0 or rr >= burnable.shape[0] or cc < 0 or cc >= burnable.shape[1]:
            return True
        if (rr, cc) in {(r, c), (nr, nc)}:
            continue
        if not burnable[rr, cc]:
            return True
    return False



def simulate_spread(
    env: dict,
    ignition_lon: float,
    ignition_lat: float,
    horizon_minutes: int,
    wind_speed_kmh: float,
    wind_direction_from_deg: float,
    fine_fuel_moisture_pct: float,
    control_lines_geojson: list[dict] | None = None,
    control_line_width_m: float = 60.0,
    control_effectiveness_pct: float = 100.0,
    control_mode: str = "hard",
) -> dict:
    inside = env["inside"]
    base = env["base_ros"]
    barrier_geom = control_barrier_geometry(env, control_lines_geojson, control_line_width_m)
    barrier = control_mask_from_lines(env, control_lines_geojson, control_line_width_m)
    hard_barrier = str(control_mode).lower() in {"hard", "barrier", "complete", "impermeable"}
    landscape_burnable = inside & (base > 0)
    burnable = landscape_burnable.copy()
    if hard_barrier and barrier.any():
        burnable = burnable & (~barrier)
    moisture_factor = _scenario_moisture_factor(fine_fuel_moisture_pct)

    r0, c0 = _ignition_cell(env, ignition_lon, ignition_lat, burnable)
    arrival = np.full((env["nrows"], env["ncols"]), np.inf, dtype="float64")
    arrival[r0, c0] = 0.0
    pq: list[tuple[float, int, int]] = [(0.0, r0, c0)]

    # Wind direction is meteorological FROM; head-fire spread is favored downwind TO.
    wind_to = (float(wind_direction_from_deg) + 180.0) % 360.0
    control_eff = float(np.clip(control_effectiveness_pct / 100.0, 0.0, 1.0))
    max_ros_seen = 0.0
    blocked_control_edges = 0
    penalized_control_edges = 0
    blocked_nonfuel_edges = 0
    min_slope_factor_seen = float("inf")
    max_slope_factor_seen = 0.0

    # Sixteen propagation directions reduce the artificial 8-neighbour
    # octagon and allow wind/slope/fuel heterogeneity to shape the front.
    # Only primitive offsets are used so the same direction is not duplicated.
    neighbors = [
        (-2, -1), (-2, 1), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
        (0, -1), (0, 1),
        (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (2, -1), (2, 1),
    ]

    while pq:
        tcur, r, c = heapq.heappop(pq)
        # Do not use exact equality here. Heap times are Python float64 while
        # raster arrays may have been serialized/reloaded with finite precision.
        # Exact comparison was the v10 bug that stopped propagation after the
        # ignition cell plus its first eight neighbours.
        if tcur > float(arrival[r, c]) + 1e-9:
            continue
        if tcur > horizon_minutes:
            break

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= env["nrows"] or nc < 0 or nc >= env["ncols"]:
                continue
            if not burnable[nr, nc]:
                if hard_barrier and inside[nr, nc] and base[nr, nc] > 0 and barrier[nr, nc]:
                    blocked_control_edges += 1
                continue

            # Do not allow the longer 16-direction links to leap across a road/
            # built-up/water/no-data cell. The edge must have continuous
            # landscape fuel between origin and destination.
            if _edge_crosses_nonfuel(landscape_burnable, r, c, nr, nc):
                blocked_nonfuel_edges += 1
                continue

            dist = env["cell_size_m"] * math.hypot(float(dr), float(dc))
            bearing = _bearing_from_delta(dc, dr)
            align = math.cos(math.radians(_angle_diff_deg(bearing, wind_to)))

            # Wind effect: directional and bounded. This is a scenario multiplier,
            # not a direct Rothermel wind-factor implementation.
            wind_factor = float(np.clip(math.exp(0.030 * float(wind_speed_kmh) * align), 0.40, 3.4))

            # Positive elevation change means spread upslope, which is favored.
            dz = float(env["dem"][nr, nc] - env["dem"][r, c])
            directional_slope = dz / max(dist, 1e-6)
            slope_factor = float(np.clip(math.exp(4.2 * directional_slope), 0.38, 3.8))
            min_slope_factor_seen = min(min_slope_factor_seen, slope_factor)
            max_slope_factor_seen = max(max_slope_factor_seen, slope_factor)

            b0 = float(base[r, c])
            b1 = float(base[nr, nc])
            # Harmonic mean makes the slower fuel on an edge matter more than
            # it did in v12's arithmetic mean.
            local_base = 2.0 * b0 * b1 / max(b0 + b1, 1e-9)

            bio_factor = math.sqrt(
                max(float(env["biomass_factor"][r, c]), 0.01)
                * max(float(env["biomass_factor"][nr, nc]), 0.01)
            )
            rate = local_base * moisture_factor * wind_factor * slope_factor * bio_factor

            # Exact vector intersection with the buffered control strip. This
            # prevents 16-direction long links from stepping over a 100–200 m
            # control line between raster-cell centres.
            crossing_m = _barrier_intersection_length(env, barrier_geom, r, c, nr, nc)
            if crossing_m > 1e-6:
                if hard_barrier or control_eff >= 0.999:
                    blocked_control_edges += 1
                    continue
                penalized_control_edges += 1

            klass0 = int(env["worldcover"][r, c])
            klass1 = int(env["worldcover"][nr, nc])
            class_cap = min(
                float(MAX_ROS_M_MIN_BY_CLASS.get(klass0, 4.0)),
                float(MAX_ROS_M_MIN_BY_CLASS.get(klass1, 4.0)),
            )
            rate = float(np.clip(rate, 0.01, max(class_cap, 0.01)))
            max_ros_seen = max(max_ros_seen, rate)
            dt = dist / rate
            if crossing_m > 1e-6 and not hard_barrier:
                # Replace the normal travel time through the intersected part
                # with a much slower crossing time. At 90% effectiveness, for
                # example, a 200 m strip imposes roughly tenfold travel time
                # through the strip instead of a cosmetic per-cell slowdown.
                crossing_rate = max(0.02, rate * max(0.01, 1.0 - control_eff))
                dt += crossing_m / crossing_rate - crossing_m / rate
            nt = tcur + dt
            if nt + 1e-9 < float(arrival[nr, nc]) and nt <= horizon_minutes:
                arrival[nr, nc] = nt
                heapq.heappush(pq, (nt, nr, nc))

    reached = np.isfinite(arrival) & (arrival <= horizon_minutes)
    cell_area_ha = (env["cell_size_m"] ** 2) / 10_000.0
    area_ha = float(reached.sum() * cell_area_ha)

    # Head bearing estimated from farthest reached cell relative to ignition.
    head_bearing = wind_to
    max_distance_m = 0.0
    head_cell = (r0, c0)
    reached_idx = np.argwhere(reached)
    if len(reached_idx):
        drs = reached_idx[:, 0] - r0
        dcs = reached_idx[:, 1] - c0
        dists = np.sqrt(drs ** 2 + dcs ** 2) * env["cell_size_m"]
        imax = int(np.argmax(dists))
        max_distance_m = float(dists[imax])
        if max_distance_m > 0:
            head_bearing = _bearing_from_delta(int(dcs[imax]), int(drs[imax]))
            head_cell = (int(reached_idx[imax, 0]), int(reached_idx[imax, 1]))

    by_time = {}
    for mins in [15, 30, 60, 120, 180, 240, 360, 720, 1440]:
        if mins <= horizon_minutes:
            by_time[str(mins)] = float(((arrival <= mins) & np.isfinite(arrival)).sum() * cell_area_ha)

    # Did the simulated front touch the AOI edge?
    boundary_touch = False
    for rr, cc in np.argwhere(reached):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = rr + dr, cc + dc
            if nr < 0 or nr >= env["nrows"] or nc < 0 or nc >= env["ncols"] or not inside[nr, nc]:
                boundary_touch = True
                break
        if boundary_touch:
            break

    return {
        "arrival_min": arrival,
        "barrier_mask": barrier,
        "ignition_cell": (r0, c0),
        "head_cell": head_cell,
        "reaches_aoi_boundary": bool(boundary_touch),
        "area_ha": area_ha,
        "reached_cells": int(reached.sum()),
        "max_distance_m": max_distance_m,
        "head_bearing_deg": float(head_bearing),
        "max_ros_m_min": float(max_ros_seen),
        "mean_slope_deg": float(np.mean(env["slope_deg"][inside])) if inside.any() else 0.0,
        "by_time_ha": by_time,
        "horizon_minutes": int(horizon_minutes),
        "wind_to_deg": float(wind_to),
        "fine_fuel_moisture_pct": float(fine_fuel_moisture_pct),
        "control_effectiveness_pct": float(control_effectiveness_pct),
        "control_line_width_m": float(control_line_width_m),
        "control_mode": "hard" if hard_barrier else "partial",
        "barrier_cells": int(barrier.sum()),
        "blocked_control_edges": int(blocked_control_edges),
        "penalized_control_edges": int(penalized_control_edges),
        "blocked_nonfuel_edges": int(blocked_nonfuel_edges),
        "min_slope_factor_seen": float(min_slope_factor_seen if np.isfinite(min_slope_factor_seen) else 1.0),
        "max_slope_factor_seen": float(max_slope_factor_seen),
        "mean_reached_biomass_factor": float(np.mean(env["biomass_factor"][reached])) if reached.any() else 1.0,
        "mean_reached_slope_deg": float(np.mean(env["slope_deg"][reached])) if reached.any() else 0.0,
        "data_quality": env.get("data_quality", {}),
    }



def _arrival_hex(ratio: float) -> str:
    """Yellow -> orange -> red -> purple ramp for arrival-time animation."""
    stops = np.array([0.0, 0.35, 0.70, 1.0], dtype="float64")
    colors = np.array([
        [255, 215, 64],
        [244, 128, 40],
        [190, 45, 45],
        [103, 45, 120],
    ], dtype="float64")
    ratio = float(np.clip(ratio, 0.0, 1.0))
    rgb = [int(round(np.interp(ratio, stops, colors[:, b]))) for b in range(3)]
    return "#%02x%02x%02x" % tuple(rgb)


def arrival_animation_geojson(
    env: dict,
    arrival_min: np.ndarray,
    horizon_minutes: int,
    frame_step_minutes: int = 30,
) -> dict:
    """Convert reached grid cells to cumulative TimestampedGeoJson features.

    Cells are grouped into time bins as MultiPolygons. With TimestampedGeoJson
    duration=None, earlier bins remain visible while later bins are added, giving
    a true cumulative fire-spread animation with a built-in Play button.
    """
    arr = np.asarray(arrival_min, dtype="float64")
    valid = np.isfinite(arr) & (arr <= float(horizon_minutes)) & env["inside"]
    if not valid.any():
        return {"type": "FeatureCollection", "features": []}

    step = max(5, int(frame_step_minutes))
    reached = np.argwhere(valid)
    bins: dict[int, list[tuple[int, int]]] = {}
    for r, c in reached:
        at = max(0.0, float(arr[int(r), int(c)]))
        frame = int(math.ceil(at / step) * step) if at > 0 else 0
        frame = min(frame, int(horizon_minutes))
        bins.setdefault(frame, []).append((int(r), int(c)))

    transformer = Transformer.from_crs(env["epsg"], 4326, always_xy=True)
    base_time = datetime(2000, 1, 1, 0, 0, tzinfo=timezone.utc)
    cell = float(env["cell_size_m"])
    features = []

    for frame in sorted(bins):
        multipoly = []
        for r, c in bins[frame]:
            x0 = env["minx"] + c * cell
            x1 = x0 + cell
            y1 = env["maxy"] - r * cell
            y0 = y1 - cell
            xs = [x0, x1, x1, x0, x0]
            ys = [y0, y0, y1, y1, y0]
            lons, lats = transformer.transform(xs, ys)
            ring = [[float(lo), float(la)] for lo, la in zip(lons, lats)]
            multipoly.append([ring])

        stamp = (base_time + timedelta(minutes=int(frame))).isoformat().replace('+00:00', 'Z')
        ratio = frame / max(float(horizon_minutes), 1.0)
        features.append({
            "type": "Feature",
            "geometry": {"type": "MultiPolygon", "coordinates": multipoly},
            "properties": {
                "times": [stamp] * len(multipoly),
                "style": {
                    "color": _arrival_hex(ratio),
                    "fillColor": _arrival_hex(ratio),
                    "weight": 0,
                    "fillOpacity": 0.68,
                },
                "popup": f"Arrival bin: {frame} min",
            },
        })

    return {"type": "FeatureCollection", "features": features}


def arrival_overlay_png(env: dict, arrival_min: np.ndarray, display_minutes: float, barrier_mask: np.ndarray | None = None) -> bytes:
    """Create a transparent arrival-time overlay for the simulator map."""
    arr = np.asarray(arrival_min)
    reached = np.isfinite(arr) & (arr <= display_minutes) & env["inside"]
    rgba = np.zeros((env["nrows"], env["ncols"], 4), dtype=np.uint8)

    if reached.any():
        ratio = np.clip(arr / max(float(display_minutes), 1.0), 0, 1)
        # Early arrival: bright yellow/orange; later arrival: dark red/purple.
        stops = np.array([0.0, 0.35, 0.70, 1.0], dtype="float32")
        colors = np.array([
            [255, 236, 80],
            [255, 145, 40],
            [210, 55, 35],
            [92, 30, 90],
        ], dtype="float32")
        for band in range(3):
            rgba[..., band] = np.interp(ratio, stops, colors[:, band]).astype(np.uint8)
        rgba[..., 3] = np.where(reached, 185, 0).astype(np.uint8)

    if barrier_mask is not None and barrier_mask.any():
        rgba[barrier_mask, 0:3] = np.array([30, 120, 220], dtype=np.uint8)
        rgba[barrier_mask, 3] = 220

    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").resize(
        (env["ncols"] * 4, env["nrows"] * 4), resample=Image.Resampling.NEAREST
    ).save(buf, format="PNG", optimize=True)
  

    return buf.getvalue()

WORLD_COVER_VIDEO_COLORS = {
    10: (95, 153, 92),
    20: (188, 160, 83),
    30: (205, 192, 100),
    40: (196, 158, 157),
    50: (190, 190, 190),
    60: (205, 195, 175),
    70: (238, 238, 238),
    80: (105, 160, 190),
    90: (100, 170, 160),
    95: (70, 145, 110),
    100: (170, 165, 115),
    0: (185, 185, 180),
}


def _spread_frame_image(env: dict, sim: dict, display_minutes: int, width_px: int = 960, height_px: int = 720) -> Image.Image:
    """Render a self-contained animation frame from the model grid.

    The export intentionally uses the analysed WorldCover grid rather than
    screenshotting OpenStreetMap tiles, so it is reproducible and does not
    depend on browser automation or external tile licensing in the video file.
    """
    rows, cols = env["nrows"], env["ncols"]
    base_img = np.zeros((rows, cols, 3), dtype=np.uint8)
    wc = env["worldcover"]
    for code, rgb in WORLD_COVER_VIDEO_COLORS.items():
        base_img[wc == code] = np.asarray(rgb, dtype=np.uint8)

    # Make the two extra model inputs visible in the exported animation.
    # 1) Woody WorldCover cells are modulated by the ESA CCI biomass factor.
    woody = env["inside"] & np.isin(wc, [10, 20])
    if woody.any():
        bf = np.asarray(env.get("biomass_factor", np.ones_like(wc, dtype=float)), dtype=float)
        # Low woody biomass = paler background; medium/high biomass = darker.
        bshade = np.clip(0.72 + 0.36 * bf, 0.72, 1.08)
        base_img[woody] = np.clip(
            base_img[woody].astype(float) * bshade[woody, None],
            0, 255
        ).astype(np.uint8)

    # 2) Copernicus DEM hillshade gives visual relief consistent with the
    # directional slope term used in travel-time calculation.
    hillshade = np.asarray(env.get("hillshade", np.ones_like(wc, dtype=float)), dtype=float)
    base_img[env["inside"]] = np.clip(
        base_img[env["inside"]].astype(float) * hillshade[env["inside"], None],
        0, 255
    ).astype(np.uint8)

    base_img[~env["inside"]] = np.asarray([245, 245, 242], dtype=np.uint8)

    arr = np.asarray(sim["arrival_min"], dtype="float64")
    reached = np.isfinite(arr) & (arr <= float(display_minutes)) & env["inside"]
    if reached.any():
        ratio = np.clip(arr / max(float(sim["horizon_minutes"]), 1.0), 0.0, 1.0)
        stops = np.array([0.0, 0.35, 0.70, 1.0])
        colors = np.array([[255, 215, 64], [244, 128, 40], [190, 45, 45], [103, 45, 120]], dtype=float)
        fire_rgb = np.zeros_like(base_img)
        for b in range(3):
            fire_rgb[..., b] = np.interp(ratio, stops, colors[:, b]).astype(np.uint8)
        base_img[reached] = (0.28 * base_img[reached] + 0.72 * fire_rgb[reached]).astype(np.uint8)

    barrier = np.asarray(sim.get("barrier_mask", np.zeros((rows, cols), dtype=bool)))
    if barrier.any():
        base_img[barrier] = np.asarray([38, 113, 205], dtype=np.uint8)

    # Scale the raster with nearest-neighbour resampling so model cells remain visible.
    map_h = max(320, height_px - 90)
    img = Image.fromarray(base_img, mode="RGB").resize((width_px, map_h), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (width_px, height_px), (26, 28, 31))
    canvas.paste(img, (0, 90))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    cell_area_ha = (env["cell_size_m"] ** 2) / 10_000.0
    area = float(reached.sum() * cell_area_ha)
    hh, mm = divmod(int(display_minutes), 60)
    title = f"PREPARE WP4 - Fire spread scenario   T+{hh:02d}:{mm:02d}"
    subtitle = (
        f"Modelled area: {area:,.1f} ha | grid: {env['cell_size_m']} m | "
        "WorldCover + CCI AGB + DEM slope | blue = control barrier"
    )
    draw.text((18, 18), title, fill=(255, 255, 255), font=font)
    draw.text((18, 48), subtitle, fill=(220, 220, 220), font=font)

    # Ignition marker in raster coordinates after scaling.
    r0, c0 = sim["ignition_cell"]
    sx = width_px / max(cols, 1)
    sy = map_h / max(rows, 1)
    x = int((c0 + 0.5) * sx)
    y = int(90 + (r0 + 0.5) * sy)
    rad = 7
    draw.ellipse((x-rad, y-rad, x+rad, y+rad), fill=(220, 35, 25), outline=(20, 20, 20), width=2)
    return canvas


def spread_animation_frames(env: dict, sim: dict, frame_step_minutes: int = 30) -> list[Image.Image]:
    step = max(5, int(frame_step_minutes))
    horizon = int(sim["horizon_minutes"])
    times = list(range(0, horizon + 1, step))
    if not times or times[-1] != horizon:
        times.append(horizon)
    return [_spread_frame_image(env, sim, minute) for minute in times]


def export_spread_gif(env: dict, sim: dict, frame_step_minutes: int = 30, fps: int = 4) -> bytes:
    frames = spread_animation_frames(env, sim, frame_step_minutes)
    if not frames:
        raise RuntimeError("No frames available for GIF export.")
    buf = io.BytesIO()
    duration_ms = max(80, int(round(1000 / max(int(fps), 1))))
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return buf.getvalue()


def export_spread_mp4(env: dict, sim: dict, frame_step_minutes: int = 30, fps: int = 4) -> bytes:
    """Export the cumulative spread animation to H.264 MP4 using imageio-ffmpeg."""
    import os
    import tempfile
    import imageio.v2 as imageio

    frames = spread_animation_frames(env, sim, frame_step_minutes)
    if not frames:
        raise RuntimeError("No frames available for MP4 export.")

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        writer = imageio.get_writer(
            path,
            fps=max(int(fps), 1),
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=1,
        )
        try:
            for frame in frames:
                writer.append_data(np.asarray(frame.convert("RGB")))
        finally:
            writer.close()
        return Path(path).read_bytes()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def ignition_cell_lonlat(env: dict, row: int, col: int) -> tuple[float, float]:
    x = env["minx"] + (col + 0.5) * env["cell_size_m"]
    y = env["maxy"] - (row + 0.5) * env["cell_size_m"]
    transformer = Transformer.from_crs(env["epsg"], 4326, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return float(lon), float(lat)


def bearing_label(deg: float) -> str:
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return labels[int(((deg % 360) + 22.5) // 45) % 8]
