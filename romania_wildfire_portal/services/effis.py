from __future__ import annotations

import io
import os
import tempfile
import zipfile
from datetime import date, datetime, timedelta

import geopandas as gpd
import numpy as np
import requests
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon

EFFIS_WMS = "https://maps.effis.emergency.copernicus.eu/effis"
GWIS_WMS = "https://maps.effis.emergency.copernicus.eu/gwis"
EFFIS_WFS = EFFIS_WMS

EFFIS_FWI_LAYER = "ecmwf.fwi"

# Exact operational GWIS layer IDs supplied by the EFFIS/GWIS viewer.
# These are intentionally not translated to the legacy ecmwf007.* names.
EFFIS_FORECAST_LAYERS = {
    "FWI": "ecmwf.fwi",
    "FFMC": "ecmwf.ffmc",
    "DC": "ecmwf.dc",
    "ISI": "ecmwf.isi",
    "BUI": "ecmwf.bui",
    "NFDRS_IC": "ecmwf.nfdrs.ic",
    "NFDRS_ROS": "ecmwf.nfdrs.ros",
}

EFFIS_CONTEXT_PRODUCTS = {
    "FUEL": {"endpoint": EFFIS_WMS, "layer": "fuel_map"},
    "MODIS_HS": {"endpoint": GWIS_WMS, "layer": "modis.hs"},
}

EFFIS_FORECAST_INFO = {
    "FWI": {"ro": "Fire Weather Index: indicator general al pericolului meteo de incendiu și al intensității potențiale.", "en": "Fire Weather Index: overall fire-weather danger indicator and proxy for potential fire intensity."},
    "FFMC": {"ro": "Fine Fuel Moisture Code: uscarea combustibililor fini de suprafață și ușurința aprinderii.", "en": "Fine Fuel Moisture Code: dryness of fine surface fuels and ease of ignition."},
    "DC": {"ro": "Drought Code: uscarea pe termen lung a stratului organic profund și efectul secetei.", "en": "Drought Code: long-term drying of deep organic layers and drought effects."},
    "ISI": {"ro": "Initial Spread Index: potențialul de propagare inițială, influențat în special de vânt și umiditatea combustibilului fin.", "en": "Initial Spread Index: initial spread potential, driven especially by wind and fine-fuel moisture."},
    "BUI": {"ro": "Build Up Index: indicator al cantității de combustibil disponibil pentru ardere.", "en": "Build Up Index: indicator of the amount of fuel available for combustion."},
    "NFDRS_IC": {"ro": "NFDRS Ignition Component: probabilitatea relativă ca o sursă de aprindere să producă un incendiu care necesită intervenție.", "en": "NFDRS Ignition Component: relative probability that a firebrand will produce a fire requiring suppression action."},
    "NFDRS_ROS": {"ro": "NFDRS Rate of Spread: indicator al vitezei potențiale de propagare a incendiului.", "en": "NFDRS Rate of Spread: indicator of potential fire spread rate."},
    "FUEL": {"ro": "Harta europeană a combustibililor EFFIS: clase de combustibil pentru comportamentul potențial al focului.", "en": "EFFIS European Fuel Map: fuel classes representing potential fire behaviour."},
    "MODIS_HS": {"ro": "Hotspot-uri MODIS publicate prin GWIS/EFFIS pentru intervalul sezonului curent.", "en": "MODIS hotspots published through GWIS/EFFIS for the current fire-season interval."},
}


EFFIS_CLASS_THRESHOLDS = [
    {"Class": "Low", "FWI": "< 11.2", "FFMC": "< 82.7", "DMC": "< 15.7", "DC": "< 256.1", "ISI": "< 3.2", "BUI": "< 24.2", "Anomaly": "< 0.0", "Ranking": "< 80.0"},
    {"Class": "Moderate", "FWI": "11.2–21.3", "FFMC": "82.7–86.1", "DMC": "15.7–27.9", "DC": "256.1–334.1", "ISI": "3.2–5.0", "BUI": "24.2–40.7", "Anomaly": "0.0–0.5", "Ranking": "80.0–90.0"},
    {"Class": "High", "FWI": "21.3–38.0", "FFMC": "86.1–89.2", "DMC": "27.9–53.1", "DC": "334.1–450.6", "ISI": "5.0–7.5", "BUI": "40.7–73.3", "Anomaly": "0.5–1.0", "Ranking": "90.0–95.0"},
    {"Class": "Very High", "FWI": "38.0–50.0", "FFMC": "89.2–93.0", "DMC": "53.1–83.6", "DC": "450.6–600.0", "ISI": "7.5–13.4", "BUI": "73.3–133.1", "Anomaly": "1.0–1.5", "Ranking": "95.0–98.0"},
    {"Class": "Extreme", "FWI": "50.0–70.0", "FFMC": "93.0–96.0", "DMC": "83.6–160.7", "DC": "600.0–749.4", "ISI": "13.4–26.8", "BUI": "133.1–193.1", "Anomaly": "1.5–2.5", "Ranking": "98.0–99.0"},
    {"Class": "Very Extreme", "FWI": "> 70.0", "FFMC": "> 96.0", "DMC": "> 160.7", "DC": "> 749.4", "ISI": "> 26.8", "BUI": "> 193.1", "Anomaly": "> 2.5", "Ranking": "> 99.0"},
]


def _as_date(day: date | str) -> date:
    if isinstance(day, date):
        return day
    return datetime.strptime(str(day)[:10], "%Y-%m-%d").date()


def fwi_wms_options(day: date | str) -> dict:
    d = _as_date(day)
    return {"url": GWIS_WMS, "layers": EFFIS_FWI_LAYER, "time": d.isoformat()}


def fetch_burned_areas(aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    west, south, east, north = aoi.total_bounds
    params = {
        "service": "WFS", "version": "1.1.0", "request": "GetFeature",
        "typeName": "ms:modis.ba.poly", "outputformat": "SHAPEZIP",
        "bbox": f"{west},{south},{east},{north},EPSG:4326",
    }
    r = requests.get(EFFIS_WFS, params=params, timeout=75)
    r.raise_for_status()
    if not r.content.startswith(b"PK"):
        raise RuntimeError("EFFIS did not return a ZIP shapefile for burned areas.")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(td)
        shp = next((os.path.join(td, f) for f in os.listdir(td) if f.lower().endswith(".shp")), None)
        if shp is None:
            raise RuntimeError("No shapefile found in EFFIS response.")
        gdf = gpd.read_file(shp).to_crs(4326)
    if gdf.empty:
        return gdf
    geom = aoi.geometry.iloc[0]
    return gdf[gdf.geometry.intersects(geom)].copy()


def summarize_burned_areas(burned: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame) -> dict:
    if burned.empty:
        return {"burned_area_ha": 0.0, "fire_perimeters": 0}
    clipped = gpd.overlay(burned, aoi[["geometry"]], how="intersection", keep_geom_type=False)
    if clipped.empty:
        return {"burned_area_ha": 0.0, "fire_perimeters": 0}
    area = clipped.to_crs(3035).area.sum() / 10_000.0
    return {"burned_area_ha": float(area), "fire_perimeters": int(len(burned))}


def _regional_context_bounds(aoi: gpd.GeoDataFrame, min_lon_span: float = 1.8, min_lat_span: float = 1.5, padding_factor: float = 1.8):
    geom = aoi.to_crs(4326).geometry.iloc[0]
    west, south, east, north = geom.bounds
    cx, cy = geom.centroid.x, geom.centroid.y
    span_x = max((east - west) * padding_factor, min_lon_span)
    span_y = max((north - south) * padding_factor, min_lat_span)
    return (
        max(-24.75, cx - span_x / 2), max(27.0, cy - span_y / 2),
        min(45.0, cx + span_x / 2), min(72.0, cy + span_y / 2),
    )


def _coord_to_px(x: float, y: float, bounds, width: int, height: int):
    west, south, east, north = bounds
    px = int(round((x - west) / max(1e-12, east - west) * (width - 1)))
    py = int(round((north - y) / max(1e-12, north - south) * (height - 1)))
    return px, py


def _draw_geometry_outline(img: Image.Image, aoi: gpd.GeoDataFrame, bounds):
    draw = ImageDraw.Draw(img)
    geom = aoi.to_crs(4326).geometry.iloc[0]
    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms) if isinstance(geom, MultiPolygon) else []
    for poly in polys:
        pts = [_coord_to_px(x, y, bounds, img.width, img.height) for x, y in poly.exterior.coords]
        if len(pts) >= 2:
            draw.line(pts, fill=(15, 15, 15, 255), width=5)
            draw.line(pts, fill=(255, 255, 255, 255), width=2)
    return img


def _valid_wms_image(content: bytes, content_type: str):
    if "image" not in (content_type or "").lower():
        sample = content[:240].decode("utf-8", errors="ignore").replace("\n", " ")
        raise RuntimeError(f"WMS returned {content_type or 'non-image content'}: {sample[:180]}")
    try:
        img = Image.open(io.BytesIO(content)).convert("RGBA")
    except Exception as exc:
        raise RuntimeError(f"Could not decode WMS image: {exc}") from exc
    arr = np.asarray(img)
    alpha = arr[..., 3]
    visible = alpha > 0
    if visible.sum() < max(20, int(alpha.size * 0.00005)):
        raise RuntimeError("WMS image is effectively empty/transparent.")
    rgb = arr[..., :3][visible]
    if rgb.size and np.ptp(rgb.astype(np.int16), axis=0).max() < 2:
        raise RuntimeError("WMS image contains no visible thematic variation.")
    return img


def _request_wms_image(endpoint: str, layer: str, bounds, time_value: str | None, width: int = 720, height: int = 540):
    west, south, east, north = bounds
    params = {
        "LAYERS": layer, "FORMAT": "image/png", "TRANSPARENT": "true",
        "SINGLETILE": "false", "SERVICE": "wms", "VERSION": "1.1.1",
        "REQUEST": "GetMap", "STYLES": "", "SRS": "EPSG:4326",
        "BBOX": f"{west},{south},{east},{north}", "WIDTH": str(width), "HEIGHT": str(height),
    }
    if time_value:
        params["TIME"] = time_value
    headers = {
        "User-Agent": "PREPARE-Romania-Wildfire-Portal/8.0",
        "Accept": "image/png,image/*;q=0.9,*/*;q=0.1",
        "Referer": "https://forest-fire.emergency.copernicus.eu/",
    }
    r = requests.get(endpoint, params=params, headers=headers, timeout=(12, 35))
    r.raise_for_status()
    return _valid_wms_image(r.content, r.headers.get("content-type", ""))


def _candidate_times(product_key: str, selected_day: date):
    if product_key == "FUEL":
        return [f"{(selected_day - timedelta(days=1)).isoformat()}/{selected_day.isoformat()}", None]
    if product_key == "MODIS_HS":
        return [f"{selected_day.year}-01-01/{selected_day.isoformat()}"]
    # Operational GWIS fire-weather links use one exact forecast day.
    # Try the selected day first, then the two previous days as availability fallbacks.
    return [(selected_day - timedelta(days=lag)).isoformat() for lag in range(0, 3)]



# Exact categorical colours used by the operational GWIS fire-danger maps.
# They were verified against the rendered GWIS products used by the portal.
# The order follows the EFFIS danger scale from Low to Very Extreme.
GWIS_DANGER_PALETTE = {
    "Low": (196, 252, 195),
    "Moderate": (210, 224, 95),
    "High": (211, 173, 49),
    "Very High": (189, 117, 42),
    "Extreme": (143, 33, 27),
    "Very Extreme": (47, 5, 21),
}
GWIS_DANGER_ORDER = ["Low", "Moderate", "High", "Very High", "Extreme", "Very Extreme"]


def _polygon_pixel_mask(aoi: gpd.GeoDataFrame, bounds, width: int, height: int) -> np.ndarray:
    """Rasterize the AOI polygon into the already-rendered WMS image space."""
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    geom = aoi.to_crs(4326).geometry.iloc[0]
    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms) if isinstance(geom, MultiPolygon) else []
    for poly in polys:
        exterior = [_coord_to_px(x, y, bounds, width, height) for x, y in poly.exterior.coords]
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=255)
        for ring in poly.interiors:
            hole = [_coord_to_px(x, y, bounds, width, height) for x, y in ring.coords]
            if len(hole) >= 3:
                draw.polygon(hole, fill=0)
    return np.asarray(mask_img) > 0


def _classify_aoi_map_colours(img: Image.Image, aoi: gpd.GeoDataFrame, bounds) -> dict:
    """Classify rendered GWIS pixels inside the AOI using the service's danger palette.

    This reads the same categorical colours the user sees in the map. It therefore
    also works for NFDRS layers where EFFIS does not publish the Canadian-FWI numeric
    breakpoints used for FWI/FFMC/DC/ISI/BUI.
    """
    rgba = np.asarray(img.convert("RGBA"))
    mask = _polygon_pixel_mask(aoi, bounds, img.width, img.height)
    valid = mask & (rgba[..., 3] > 0)
    if valid.sum() == 0:
        return {"danger_class": None, "danger_class_max": None, "danger_class_share_pct": None, "danger_class_counts": {}}

    pixels = rgba[..., :3][valid].astype(np.int32)
    palette_names = list(GWIS_DANGER_PALETTE.keys())
    palette = np.asarray([GWIS_DANGER_PALETTE[k] for k in palette_names], dtype=np.int32)
    # Nearest palette colour. A generous threshold admits antialiased/scaled pixels
    # while rejecting white/transparent/background colours and AOI linework.
    d2 = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1)
    nearest_d2 = d2[np.arange(d2.shape[0]), nearest]
    accepted = nearest_d2 <= (70 ** 2)
    if accepted.sum() == 0:
        return {"danger_class": None, "danger_class_max": None, "danger_class_share_pct": None, "danger_class_counts": {}}

    nearest = nearest[accepted]
    counts = {name: int((nearest == i).sum()) for i, name in enumerate(palette_names)}
    counts = {k: v for k, v in counts.items() if v > 0}
    dominant = max(counts, key=counts.get)
    total = sum(counts.values())
    dominant_share = 100.0 * counts[dominant] / total if total else None

    # Ignore tiny edge/antialias fragments when reporting the maximum intersecting class.
    meaningful = [
        name for name in GWIS_DANGER_ORDER
        if counts.get(name, 0) >= max(3, int(total * 0.01))
    ]
    max_class = meaningful[-1] if meaningful else dominant
    return {
        "danger_class": dominant,
        "danger_class_max": max_class,
        "danger_class_share_pct": float(dominant_share) if dominant_share is not None else None,
        "danger_class_counts": counts,
        "danger_class_method": "GWIS rendered danger colours inside AOI",
    }

def effis_product_thumbnail(aoi: gpd.GeoDataFrame, day: date | str, product_key: str, width: int = 720, height: int = 540):
    """Download and validate a regional EFFIS/GWIS WMS image, then draw the AOI locally."""
    aoi = aoi.to_crs(4326)
    selected_day = _as_date(day)
    bounds = _regional_context_bounds(aoi)
    if product_key in EFFIS_FORECAST_LAYERS:
        endpoint = GWIS_WMS; layer = EFFIS_FORECAST_LAYERS[product_key]
    elif product_key in EFFIS_CONTEXT_PRODUCTS:
        endpoint = EFFIS_CONTEXT_PRODUCTS[product_key]["endpoint"]
        layer = EFFIS_CONTEXT_PRODUCTS[product_key]["layer"]
    else:
        raise KeyError(f"Unknown EFFIS/GWIS product: {product_key}")

    errors, img, used_time = [], None, None
    for time_value in _candidate_times(product_key, selected_day):
        try:
            img = _request_wms_image(endpoint, layer, bounds, time_value, width=width, height=height)
            used_time = time_value or "not required"
            break
        except Exception as exc:
            errors.append(f"{time_value or 'no TIME'}: {exc}")
    if img is None:
        raise RuntimeError(" | ".join(errors[:4]))

    # Classify the original thematic pixels before drawing the AOI outline.
    danger = {}
    if product_key in EFFIS_FORECAST_LAYERS:
        danger = _classify_aoi_map_colours(img, aoi, bounds)

    bg = Image.new("RGBA", img.size, (247, 247, 244, 255))
    img = Image.alpha_composite(bg, img)
    img = _draw_geometry_outline(img, aoi, bounds)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True, quality=92)
    result = {"png": buf.getvalue(), "time": used_time, "endpoint": endpoint, "layer": layer, "bounds": bounds, "product": product_key}
    result.update(danger)
    return result
