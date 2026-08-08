from __future__ import annotations

import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon


def geojson_to_gdf(geojson: dict) -> gpd.GeoDataFrame:
    geom = shape(geojson)
    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise ValueError("Please draw a polygon, not a point or line.")
    return gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs="EPSG:4326")


def area_ha(gdf: gpd.GeoDataFrame) -> float:
    return float(gdf.to_crs(3035).area.iloc[0] / 10_000.0)


def bounds_csv(gdf: gpd.GeoDataFrame) -> str:
    west, south, east, north = gdf.total_bounds
    return f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}"


def centroid_lonlat(gdf: gpd.GeoDataFrame) -> tuple[float, float]:
    c = gdf.to_crs(3035).centroid.to_crs(4326).iloc[0]
    return float(c.x), float(c.y)


def as_geojson_geometry(gdf: gpd.GeoDataFrame) -> dict:
    return mapping(gdf.geometry.iloc[0])
