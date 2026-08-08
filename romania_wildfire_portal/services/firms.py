from __future__ import annotations

from io import StringIO
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def fetch_firms(map_key: str, source: str, bbox: str, day_range: int = 2) -> gpd.GeoDataFrame:
    if not map_key:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    url = f"{FIRMS_BASE}/{map_key}/{source}/{bbox}/{int(day_range)}"
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    text = r.text.strip()
    if not text or text.startswith("Invalid") or text.startswith("Error"):
        raise RuntimeError(text or "FIRMS returned an empty response")
    df = pd.read_csv(StringIO(text))
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )
    return gdf


def filter_to_aoi(fires: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if fires.empty:
        return fires
    geom = aoi.geometry.iloc[0]
    return fires[fires.geometry.intersects(geom)].copy()


def summarize_fires(fires: gpd.GeoDataFrame) -> dict:
    if fires.empty:
        return {
            "hotspots": 0,
            "frp_sum_mw": 0.0,
            "frp_mean_mw": 0.0,
            "frp_max_mw": 0.0,
            "latest": None,
        }
    frp = pd.to_numeric(fires.get("frp", pd.Series(dtype=float)), errors="coerce")
    latest = None
    if "acq_date" in fires.columns:
        dates = pd.to_datetime(fires["acq_date"], errors="coerce")
        if dates.notna().any():
            latest = str(dates.max().date())
    return {
        "hotspots": int(len(fires)),
        "frp_sum_mw": float(frp.sum(skipna=True)) if len(frp) else 0.0,
        "frp_mean_mw": float(frp.mean(skipna=True)) if frp.notna().any() else 0.0,
        "frp_max_mw": float(frp.max(skipna=True)) if frp.notna().any() else 0.0,
        "latest": latest,
    }
