from __future__ import annotations

import heapq
import io
import math
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from PIL import Image
from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiLineString, Point, box, mapping
from shapely.ops import transform as shp_transform
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
    10: 3.5,   # tree cover
    20: 6.0,   # shrubland
    30: 10.0,  # grassland
    40: 4.5,   # cropland
    50: 0.0,   # built-up treated as non-wildland fuel
    60: 0.7,   # bare/sparse
    70: 0.0,   # snow/ice
    80: 0.0,   # water
    90: 1.0,   # herbaceous wetland
    95: 2.0,   # mangroves (not expected in Romania)
    100: 1.0,  # moss/lichen
    0: 2.5,    # unknown fallback
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
) -> tuple[np.ndarray, list[str]]:
    """Sample remote COGs efficiently by reading one bounding window per tile."""
    out = np.full(lons.shape, fallback, dtype="float32")
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
                    out[group_idxs] = vals
            except Exception as exc:
                warnings.append(f"{key}: {exc}")
    return out, warnings


def _sample_biomass(lons: np.ndarray, lats: np.ndarray, valid_flat: np.ndarray) -> tuple[np.ndarray, list[str]]:
    out = np.zeros(lons.shape, dtype="float32")
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
                for v in src.sample(coords, indexes=1, masked=True):
                    x = float(v[0]) if not np.ma.is_masked(v[0]) else 0.0
                    if not np.isfinite(x) or x < 0 or x > 5000:
                        x = 0.0
                    vals.append(x)
                out[group_idxs] = np.asarray(vals, dtype="float32")
        except Exception as exc:
            warnings.append(f"N{north_edge} E{west_edge}: {exc}")
    return out, warnings


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
    wc_flat, wc_warn = _sample_grouped_remote(
        lon_flat,
        lat_flat,
        valid_flat,
        _worldcover_tile_origin,
        _worldcover_tile_url,
        fallback=0.0,
    )
    worldcover = wc_flat.reshape(nrows, ncols).astype("int16")

    # Biomass context from ESA CCI.
    biomass_flat, bio_warn = _sample_biomass(lon_flat, lat_flat, valid_flat)
    biomass = biomass_flat.reshape(nrows, ncols)

    # Terrain from Copernicus DEM GLO-30 COGs.
    dem_flat, dem_warn = _sample_grouped_remote(
        lon_flat,
        lat_flat,
        valid_flat,
        _dem_tile_origin,
        _dem_tile_url,
        fallback=np.nan,
    )
    dem = dem_flat.reshape(nrows, ncols)
    dem = _nearest_fill(dem, inside, default=0.0)

    # Elevation gradients: northing increases upward, array rows increase southward.
    dz_dy, dz_dx = np.gradient(dem, -cell_size_m, cell_size_m)
    slope_tan = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    slope_deg = np.degrees(np.arctan(slope_tan))

    base_ros = np.zeros_like(dem, dtype="float32")
    for klass, rate in BASE_ROS_M_MIN.items():
        base_ros[worldcover == klass] = float(rate)
    base_ros[(inside) & (base_ros == 0) & (~np.isin(worldcover, [50, 70, 80]))] = BASE_ROS_M_MIN[0]
    base_ros[~inside] = 0.0

    # Biomass is used as a modest multiplier so CCI does not dominate the model.
    biomass_factor = 0.75 + 0.55 * np.clip(biomass, 0, 250) / 250.0
    biomass_factor[~inside] = 1.0

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
        "dem": dem.astype("float32"),
        "slope_deg": slope_deg.astype("float32"),
        "base_ros": base_ros.astype("float32"),
        "biomass_factor": biomass_factor.astype("float32"),
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


def control_mask_from_lines(env: dict, lines_geojson: list[dict] | None, width_m: float) -> np.ndarray:
    mask = np.zeros((env["nrows"], env["ncols"]), dtype=bool)
    if not lines_geojson:
        return mask

    line_geoms = []
    for item in lines_geojson:
        geom = item.get("geometry", item)
        gtype = geom.get("type") if isinstance(geom, dict) else None
        coords = geom.get("coordinates") if isinstance(geom, dict) else None
        if gtype == "LineString" and coords:
            line_geoms.append(LineString(coords))
        elif gtype == "MultiLineString" and coords:
            line_geoms.append(MultiLineString(coords))
    if not line_geoms:
        return mask

    transformed = [_transform_geometry_to_epsg(g, env["epsg"]) for g in line_geoms]
    barrier = transformed[0]
    for g in transformed[1:]:
        barrier = barrier.union(g)
    barrier = barrier.buffer(max(width_m, env["cell_size_m"] * 0.35) / 2.0)
    pbar = prep(barrier)

    for r in range(env["nrows"]):
        y = env["maxy"] - (r + 0.5) * env["cell_size_m"]
        for c in range(env["ncols"]):
            if not env["inside"][r, c]:
                continue
            x = env["minx"] + (c + 0.5) * env["cell_size_m"]
            mask[r, c] = pbar.contains(Point(x, y))
    return mask


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
) -> dict:
    inside = env["inside"]
    base = env["base_ros"]
    burnable = inside & (base > 0)
    barrier = control_mask_from_lines(env, control_lines_geojson, control_line_width_m)
    moisture_factor = _scenario_moisture_factor(fine_fuel_moisture_pct)

    r0, c0 = _ignition_cell(env, ignition_lon, ignition_lat, burnable)
    arrival = np.full((env["nrows"], env["ncols"]), np.inf, dtype="float32")
    arrival[r0, c0] = 0.0
    pq: list[tuple[float, int, int]] = [(0.0, r0, c0)]

    # Wind direction is meteorological FROM; head-fire spread is favored downwind TO.
    wind_to = (float(wind_direction_from_deg) + 180.0) % 360.0
    control_eff = float(np.clip(control_effectiveness_pct / 100.0, 0.0, 1.0))
    max_ros_seen = 0.0

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    while pq:
        tcur, r, c = heapq.heappop(pq)
        if tcur != float(arrival[r, c]):
            continue
        if tcur > horizon_minutes:
            break

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= env["nrows"] or nc < 0 or nc >= env["ncols"]:
                continue
            if not burnable[nr, nc]:
                continue

            dist = env["cell_size_m"] * (math.sqrt(2.0) if dr and dc else 1.0)
            bearing = _bearing_from_delta(dc, dr)
            align = math.cos(math.radians(_angle_diff_deg(bearing, wind_to)))

            # Wind effect: directional and bounded. This is a scenario multiplier,
            # not a direct Rothermel wind-factor implementation.
            wind_factor = float(np.clip(math.exp(0.045 * float(wind_speed_kmh) * align), 0.35, 3.8))

            # Positive elevation change means spread upslope, which is favored.
            dz = float(env["dem"][nr, nc] - env["dem"][r, c])
            directional_slope = dz / max(dist, 1e-6)
            slope_factor = float(np.clip(math.exp(3.5 * directional_slope), 0.50, 3.5))

            local_base = 0.5 * (float(base[r, c]) + float(base[nr, nc]))
            bio_factor = 0.5 * (float(env["biomass_factor"][r, c]) + float(env["biomass_factor"][nr, nc]))
            rate = local_base * moisture_factor * wind_factor * slope_factor * bio_factor

            if barrier[nr, nc] or barrier[r, c]:
                if control_eff >= 0.995:
                    continue
                rate *= max(0.05, 1.0 - control_eff)

            rate = float(np.clip(rate, 0.08, 70.0))
            max_ros_seen = max(max_ros_seen, rate)
            dt = dist / rate
            nt = tcur + dt
            if nt < float(arrival[nr, nc]) and nt <= horizon_minutes * 1.25:
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
    for mins in [30, 60, 120, 180, 240, 360, 720]:
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
    }


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


def ignition_cell_lonlat(env: dict, row: int, col: int) -> tuple[float, float]:
    x = env["minx"] + (col + 0.5) * env["cell_size_m"]
    y = env["maxy"] - (row + 0.5) * env["cell_size_m"]
    transformer = Transformer.from_crs(env["epsg"], 4326, always_xy=True)
    lon, lat = transformer.transform(x, y)
    return float(lon), float(lat)


def bearing_label(deg: float) -> str:
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return labels[int(((deg % 360) + 22.5) // 45) % 8]
