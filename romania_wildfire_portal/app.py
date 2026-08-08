from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from shapely.geometry import Point
from dotenv import load_dotenv
from folium.plugins import Draw, Fullscreen, MeasureControl, MousePosition, TimestampedGeoJson
from streamlit_folium import st_folium

from services.geometry import geojson_to_gdf, area_ha, bounds_csv, centroid_lonlat
from services.firms import fetch_firms, filter_to_aoi, summarize_fires
from services.effis import (
    EFFIS_FORECAST_INFO,
    EFFIS_FORECAST_LAYERS,
    EFFIS_CONTEXT_PRODUCTS,
    EFFIS_CLASS_THRESHOLDS,
    fetch_burned_areas,
    summarize_burned_areas,
    effis_product_thumbnail,
)
from services.openmeteo import fetch_weather, portal_hazard_score
from services.biomass_cci import biomass_stats, CCI_YEAR, CCI_VERSION, biomass_thumbnail
from services.worldcover import landcover_stats, worldcover_thumbnail
from services.reporting import make_html_report
from services.fire_spread import (
    build_environment,
    estimate_fine_fuel_moisture_proxy,
    simulate_spread,
    arrival_overlay_png,
    arrival_animation_geojson,
    ignition_cell_lonlat,
    bearing_label,
    BASE_ROS_M_MIN,
    FUEL_NAMES,
)

load_dotenv()

DEFAULT_FIRMS_MAP_KEY = "87f35b3f6f3784bd1eba380d16e0198b"
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", DEFAULT_FIRMS_MAP_KEY).strip() or DEFAULT_FIRMS_MAP_KEY
PROJECT_URL = "https://utcb.ro/cercetare/programe-uefiscdi/pn-iv-pro-coex-2024-1-prepare/"
MAX_AOI_HA = 5000.0  # 50 km² = 5,000 ha

st.set_page_config(page_title="PREPARE | Romania Wildfire Portal", page_icon="🔥", layout="wide")

language = st.sidebar.selectbox("Limbă / Language", ["Română", "English"], index=0)
RO = language == "Română"


def t(ro: str, en: str) -> str:
    return ro if RO else en


st.markdown(
    """
<style>
.block-container {padding-top: 1rem; max-width: 98rem;}
[data-testid="stMetric"] {background:rgba(247,247,244,.07);border:1px solid rgba(150,150,150,.25);padding:10px 14px;border-radius:12px;}
.hero {padding:16px 20px;border-radius:16px;background:linear-gradient(90deg,#3d211b,#8f2f1f);color:white;margin-bottom:10px}
.hero h1 {margin:0;font-size:2rem}.hero p{margin:.30rem 0 0 0;opacity:.92}
.project-card {border:1px solid rgba(150,150,150,.25);border-radius:14px;padding:12px 15px;margin:7px 0 12px 0;background:rgba(120,120,120,.05)}
.project-card b {color:#ef8b68}
.map-help {margin:0 0 0.5rem 0; opacity:0.82;}
.brand-strip {display:flex;gap:14px;align-items:stretch;justify-content:flex-start;flex-wrap:wrap;margin:2px 0 12px 0;}
.brand-box {display:flex;align-items:center;justify-content:center;height:92px;min-width:220px;max-width:370px;padding:10px 16px;background:white;border:1px solid rgba(150,150,150,.28);border-radius:14px;}
.brand-box img {max-height:70px;max-width:330px;width:auto;object-fit:contain;}
.brand-label {font-size:.72rem;opacity:.68;margin-top:4px;}
.thumb-card {border:1px solid rgba(150,150,150,.28);border-radius:15px;overflow:hidden;background:rgba(120,120,120,.045);height:100%;margin-bottom:8px;}
.thumb-card img {width:100%;aspect-ratio:1.35/1;object-fit:cover;display:block;background:#f6f6f4;}
.thumb-body {padding:10px 12px 12px 12px;}
.thumb-title {font-weight:750;font-size:1rem;margin-bottom:3px;}
.thumb-meta {font-size:.78rem;opacity:.70;margin-bottom:6px;}
.thumb-desc {font-size:.86rem;opacity:.90;line-height:1.35;}
.danger-result {margin-top:9px;padding:7px 9px;border-radius:9px;background:rgba(239,108,62,.10);border:1px solid rgba(239,108,62,.28);font-size:.82rem;line-height:1.35;}
.danger-result b {color:#f09a72;}
.thumb-badge {display:inline-block;padding:2px 8px;border-radius:999px;background:rgba(239,108,62,.15);color:#ef8b68;font-size:.72rem;font-weight:700;margin-bottom:6px;}
.gallery-head {display:flex;align-items:end;justify-content:space-between;gap:10px;margin-top:12px;}
.spread-box {border:1px solid rgba(239,108,62,.34);border-radius:16px;padding:14px 16px;background:rgba(239,108,62,.055);margin:10px 0 14px 0;}
.spread-box b {color:#ef8b68;}
.spread-warning {border-left:4px solid #e2a21a;padding:9px 12px;background:rgba(226,162,26,.08);border-radius:8px;margin-bottom:12px;}
</style>
""",
    unsafe_allow_html=True,
)

hero_sub = t(
    "PREPARE • WP4 Incendii de pădure și riscuri asociate • focare active • pericol de incendiu • biomasă • umiditatea solului • analize pe poligon",
    "PREPARE • WP4 Forest fires and related risks • active fires • fire danger • biomass • soil moisture • polygon analyses",
)
st.markdown(
    f'<div class="hero"><h1>🔥 {t("Portal pentru incendiile de pădure din România", "Romania Forest Wildfire Portal")}</h1><p>{hero_sub}</p></div>',
    unsafe_allow_html=True,
)

unitbv_bytes = (Path(__file__).parent / "assets" / "unitbv_official.png").read_bytes()
icas_bytes = (Path(__file__).parent / "assets" / "icas_logo.jpg").read_bytes()
unitbv_uri = "data:image/png;base64," + base64.b64encode(unitbv_bytes).decode("ascii")
icas_uri = "data:image/jpeg;base64," + base64.b64encode(icas_bytes).decode("ascii")
brand_html = f"""
<div class="brand-strip">
  <div>
    <div class="brand-box"><img src="{unitbv_uri}" alt="Universitatea Transilvania din Brașov"></div>
    <div class="brand-label">UNITBV</div>
  </div>
  <div>
    <div class="brand-box"><img src="{icas_uri}" alt="ICAS"></div>
    <div class="brand-label">ICAS</div>
  </div>
</div>
"""
st.markdown(brand_html, unsafe_allow_html=True)

project_text = t(
    "Portalul este dezvoltat ca prototip de suport pentru <b>WP4 – Incendii de pădure</b> din cadrul Centrului de Excelență <b>PREPARE</b> (PN-IV-P6-6.1-CoEx-2024-0102, 2026–2030). WP4 urmărește modelarea, monitorizarea, prevenția și evaluarea efectelor asociate incendiilor de pădure.",
    "This portal is developed as a decision-support prototype for <b>WP4 – Forest fires</b> within the <b>PREPARE</b> Centre of Excellence (PN-IV-P6-6.1-CoEx-2024-0102, 2026–2030). WP4 addresses modelling, monitoring, prevention and assessment of wildfire-related effects.",
)
st.markdown(
    f'<div class="project-card">{project_text} &nbsp; <a href="{PROJECT_URL}" target="_blank">{t("Pagina oficială PREPARE", "Official PREPARE page")} ↗</a></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header(t("Setări portal", "Portal settings"))
    st.success(t("Cheia NASA FIRMS este integrată", "NASA FIRMS key embedded"))
    source = st.selectbox(
        t("Senzor FIRMS", "FIRMS sensor"),
        ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"],
        index=0,
    )
    day_range = st.slider(t("Fereastra focarelor recente (zile)", "Recent fire window (days)"), 1, 5, 2)
    fwi_day = st.date_input(
        t("Data straturilor EFFIS", "EFFIS layer date"),
        value=date.today(),
        min_value=date.today() - timedelta(days=5),
        max_value=date.today() + timedelta(days=8),
        help=t("Folosită ca TIME=YYYY-MM-DD pentru straturile meteo GWIS ECMWF.", "Used as TIME=YYYY-MM-DD for the GWIS ECMWF fire-weather layers."),
    )
    st.divider()
    st.caption(t("Harta interactivă folosește doar OpenStreetMap și focarele active FIRMS, pentru performanță și stabilitate.", "The interactive map uses only OpenStreetMap and active FIRMS hotspots, for performance and stability."))
    st.caption(t("După analiză, portalul generează miniaturi separate pentru WorldCover, ESA CCI Biomass, harta de combustibil EFFIS, hotspot-uri GWIS MODIS și straturile GWIS ECMWF indicate de serviciul operațional.", "After analysis, the portal generates separate thumbnails for WorldCover, ESA CCI Biomass, the EFFIS fuel map, GWIS MODIS hotspots and the operational GWIS ECMWF layers."))
    st.warning(t("Limita AOI: maximum 5.000 ha (50 km²). AOI mai mari sunt blocate pentru a evita căderi de performanță.", "AOI limit: maximum 5,000 ha (50 km²). Larger AOIs are blocked to avoid crashes or severe slowdown."))
    with st.expander(t("Despre PREPARE", "About PREPARE")):
        st.markdown(t(
            "**PREPARE – Proactive Resilience and Emergency Preparedness for Adaptive Response and Efficiency** este un Centru național de Excelență pentru managementul riscului de dezastre și reziliență. Proiectul are 15 instituții partenere, 8 pachete de lucru și o perioadă de implementare 2026–2030. Acest portal sprijină activitățile WP4 privind incendiile de pădure.",
            "**PREPARE – Proactive Resilience and Emergency Preparedness for Adaptive Response and Efficiency** is a national Centre of Excellence for disaster-risk management and resilience. The project brings together 15 partner institutions, 8 work packages and runs from 2026 to 2030. This portal supports WP4 wildfire activities.",
        ))
        st.markdown(f"[{t('Pagina proiectului', 'Project page')}]({PROJECT_URL})")


@st.cache_data(ttl=900, show_spinner=False)
def cached_firms(sensor: str, bbox: str, days: int):
    return fetch_firms(FIRMS_MAP_KEY, sensor, bbox, days)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_effis_burned(geojson_str: str):
    aoi = geojson_to_gdf(json.loads(geojson_str))
    return fetch_burned_areas(aoi)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_biomass(geojson_str: str, aoi_area: float):
    aoi = geojson_to_gdf(json.loads(geojson_str))
    return biomass_stats(aoi, aoi_area_ha=aoi_area)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_worldcover(geojson_str: str, aoi_area: float):
    aoi = geojson_to_gdf(json.loads(geojson_str))
    return landcover_stats(aoi, aoi_area_ha=aoi_area)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_worldcover_thumb(geojson_str: str):
    aoi = geojson_to_gdf(json.loads(geojson_str))
    return worldcover_thumbnail(aoi)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_biomass_thumb(geojson_str: str):
    aoi = geojson_to_gdf(json.loads(geojson_str))
    return biomass_thumbnail(aoi)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_spread_environment(geojson_str: str, cell_size_m: int = 100):
    aoi = geojson_to_gdf(json.loads(geojson_str))
    return build_environment(aoi, cell_size_m=cell_size_m)


@st.cache_data(ttl=900, show_spinner=False)
def cached_effis_gallery(geojson_str: str, day_text: str):
    aoi = geojson_to_gdf(json.loads(geojson_str))
    product_order = ["FUEL", "MODIS_HS", "FWI", "FFMC", "DC", "ISI", "BUI", "NFDRS_IC", "NFDRS_ROS"]
    products, errors = {}, {}

    def fetch_one(key):
        return key, effis_product_thumbnail(aoi, day_text, key)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_one, key): key for key in product_order}
        for future in as_completed(futures):
            key = futures[future]
            try:
                _, product = future.result()
                products[key] = product
            except Exception as exc:
                errors[key] = str(exc)
    return products, errors



@st.cache_data(ttl=3600, show_spinner=False)
def cached_active_fires_thumb(geojson_str: str, sensor: str, bbox: str, days: int):
    aoi = geojson_to_gdf(json.loads(geojson_str))
    fires = cached_firms(sensor, bbox, days)
    fires_aoi = filter_to_aoi(fires, aoi)
    return active_fires_thumbnail(aoi, fires_aoi)


def build_map():
    m = folium.Map(location=[45.9, 24.9], zoom_start=7, tiles="OpenStreetMap", control_scale=True, prefer_canvas=True)
    try:
        fires = cached_firms(source, "20.0,43.4,30.0,48.5", day_range)
        fg = folium.FeatureGroup(name=f"NASA FIRMS – {source}", overlay=True, show=True)
        for _, r in fires.iterrows():
            popup = (
                f"{source}<br>{r.get('acq_date','')} {r.get('acq_time','')}"
                f"<br>FRP: {r.get('frp','n/a')} MW<br>Confidence: {r.get('confidence','n/a')}"
            )
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x], radius=3.8, color="#9e2619", fill=True,
                fill_color="#ff5a36", fill_opacity=.88, weight=1, popup=popup,
            ).add_to(fg)
        fg.add_to(m)
    except Exception as exc:
        st.sidebar.warning(t(f"Stratul FIRMS nu este disponibil: {exc}", f"FIRMS map layer unavailable: {exc}"))

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "rectangle": True,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": {"allowIntersection": False, "showArea": True},
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)
    Fullscreen().add_to(m)
    MeasureControl(primary_length_unit="kilometers", primary_area_unit="hectares").add_to(m)
    MousePosition(position="bottomright", separator=" | ", prefix="Lat/Lon:").add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    return m


def _latest_firms_seed(fires_aoi: gpd.GeoDataFrame | None):
    if not isinstance(fires_aoi, gpd.GeoDataFrame) or fires_aoi.empty:
        return None
    df = fires_aoi.copy()
    try:
        date_txt = df.get("acq_date", "").astype(str)
        time_txt = df.get("acq_time", "0000").astype(str).str.zfill(4)
        df["_dt"] = pd.to_datetime(date_txt + " " + time_txt.str[:2] + ":" + time_txt.str[2:], errors="coerce")
        row = df.sort_values("_dt").iloc[-1]
    except Exception:
        row = df.iloc[-1]
    return float(row.geometry.x), float(row.geometry.y)


def _extract_control_lines(drawings) -> list[dict]:
    lines = []
    for d in drawings or []:
        geom = d.get("geometry", {}) if isinstance(d, dict) else {}
        if geom.get("type") in {"LineString", "MultiLineString"}:
            lines.append(d)
    return lines


def _point_inside_aoi(aoi: gpd.GeoDataFrame, lon: float, lat: float) -> bool:
    try:
        return bool(aoi.to_crs(4326).geometry.iloc[0].covers(Point(float(lon), float(lat))))
    except Exception:
        return False


def _build_spread_setup_map(aoi: gpd.GeoDataFrame, seed_lon: float, seed_lat: float, existing_lines: list[dict] | None = None):
    aoi4326 = aoi.to_crs(4326)
    west, south, east, north = aoi4326.total_bounds
    center = [(south + north) / 2, (west + east) / 2]
    m = folium.Map(location=center, tiles="OpenStreetMap", control_scale=True)
    folium.GeoJson(
        aoi4326.to_json(),
        name=t("AOI selectat", "Selected AOI"),
        style_function=lambda _: {"color": "#222222", "weight": 3, "fillColor": "#ff5a36", "fillOpacity": 0.05},
    ).add_to(m)
    folium.Marker(
        [seed_lat, seed_lon],
        tooltip=t("Punct de pornire / seed curent", "Current ignition seed"),
        icon=folium.Icon(color="red", icon="fire", prefix="fa"),
    ).add_to(m)
    for line in existing_lines or []:
        geom = line.get("geometry", line) if isinstance(line, dict) else {}
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
            folium.PolyLine([(y, x) for x, y in coords], color="#2678d4", weight=5, opacity=0.95,
                            tooltip=t("Linie ipotetică de control", "Hypothetical control line")).add_to(m)
    Draw(
        export=False,
        draw_options={
            "polyline": {"shapeOptions": {"color": "#2678d4", "weight": 5}},
            "rectangle": False,
            "polygon": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)
    m.fit_bounds([[south, west], [north, east]], padding=(18, 18))
    return m


def _build_spread_result_map(aoi: gpd.GeoDataFrame, env: dict, sim: dict, display_min: int, ignition_lon: float, ignition_lat: float, control_lines: list[dict]):
    aoi4326 = aoi.to_crs(4326)
    west, south, east, north = aoi4326.total_bounds
    m = folium.Map(location=[(south+north)/2, (west+east)/2], tiles="OpenStreetMap", control_scale=True)
    png = arrival_overlay_png(env, sim["arrival_min"], display_min, sim.get("barrier_mask"))
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    folium.raster_layers.ImageOverlay(
        image=data_uri,
        bounds=env["leaflet_bounds"],
        opacity=0.82,
        interactive=False,
        cross_origin=False,
        name=t("Sosire estimată a frontului", "Estimated fire arrival"),
    ).add_to(m)
    folium.GeoJson(
        aoi4326.to_json(),
        style_function=lambda _: {"color": "#111111", "weight": 3, "fillOpacity": 0.0},
        name="AOI",
    ).add_to(m)
    folium.CircleMarker(
        [ignition_lat, ignition_lon], radius=6, color="#5c0b0b", fill=True, fill_color="#ff3d21", fill_opacity=1,
        tooltip=t("Seed incendiu", "Fire seed"),
    ).add_to(m)
    for line in control_lines or []:
        geom = line.get("geometry", line)
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
            folium.PolyLine([(y, x) for x, y in coords], color="#2678d4", weight=5, opacity=0.95,
                            tooltip=t("Linie ipotetică de control", "Hypothetical control line")).add_to(m)
    hr, hc = sim.get("head_cell", sim["ignition_cell"])
    head_lon, head_lat = ignition_cell_lonlat(env, int(hr), int(hc))
    folium.PolyLine(
        [(ignition_lat, ignition_lon), (head_lat, head_lon)],
        color="#8a1c12", weight=3, dash_array="8,6",
        tooltip=t("Direcția capului de foc modelat", "Modelled head-fire direction"),
    ).add_to(m)
    folium.Marker([head_lat, head_lon], tooltip=t("Capul frontului la orizontul modelat", "Head of front at model horizon"),
                  icon=folium.Icon(color="darkred", icon="flag", prefix="fa")).add_to(m)
    legend = f"""
    <div style="position:fixed;bottom:32px;left:55px;z-index:9999;background:rgba(255,255,255,.94);color:#222;padding:8px 10px;border-radius:7px;border:1px solid #888;font-size:11px;">
      <b>{t('Timp de sosire modelat', 'Modelled arrival time')}</b><br>
      🟨 {t('mai devreme', 'earlier')} &nbsp; 🟧 → 🟥 → 🟪 {t('mai târziu', 'later')}<br>
      <span style="color:#2678d4">━</span> {t('linie ipotetică de control', 'hypothetical control line')}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([[south, west], [north, east]], padding=(18, 18))
    Fullscreen().add_to(m)
    return m


def _build_spread_animation_map(aoi: gpd.GeoDataFrame, env: dict, sim: dict, ignition_lon: float, ignition_lat: float, control_lines: list[dict], frame_step_min: int):
    """Leaflet TimeDimension map with a cumulative Play animation."""
    aoi4326 = aoi.to_crs(4326)
    west, south, east, north = aoi4326.total_bounds
    m = folium.Map(location=[(south+north)/2, (west+east)/2], tiles="OpenStreetMap", control_scale=True)

    anim_geojson = arrival_animation_geojson(
        env, sim["arrival_min"], int(sim["horizon_minutes"]), frame_step_minutes=int(frame_step_min)
    )
    if anim_geojson.get("features"):
        TimestampedGeoJson(
            anim_geojson,
            transition_time=420,
            loop=False,
            auto_play=False,
            add_last_point=False,
            period=f"PT{int(frame_step_min)}M",
            duration=None,
            min_speed=0.25,
            max_speed=8,
            loop_button=True,
            date_options="HH:mm",
            time_slider_drag_update=True,
            speed_slider=True,
        ).add_to(m)

    folium.GeoJson(
        aoi4326.to_json(),
        style_function=lambda _: {"color": "#111111", "weight": 3, "fillOpacity": 0.0},
        name="AOI",
    ).add_to(m)
    folium.CircleMarker(
        [ignition_lat, ignition_lon], radius=6, color="#5c0b0b", fill=True, fill_color="#ff3d21", fill_opacity=1,
        tooltip=t("Seed incendiu / 00:00", "Fire seed / 00:00"),
    ).add_to(m)
    for line in control_lines or []:
        geom = line.get("geometry", line)
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
            folium.PolyLine([(y, x) for x, y in coords], color="#2678d4", weight=5, opacity=0.95,
                            tooltip=t("Linie ipotetică de control", "Hypothetical control line")).add_to(m)

    legend = f"""
    <div style="position:fixed;bottom:58px;right:12px;z-index:9999;background:rgba(255,255,255,.94);color:#222;padding:8px 10px;border-radius:7px;border:1px solid #888;font-size:11px;max-width:230px;">
      <b>{t('Animația propagării', 'Spread animation')}</b><br>
      ▶ {t('Play pornește evoluția temporală', 'Play starts temporal progression')}<br>
      00:00 = {t('momentul aprinderii', 'ignition time')}<br>
      🟨 {t('mai devreme', 'earlier')} &nbsp; 🟧 → 🟥 → 🟪 {t('mai târziu', 'later')}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([[south, west], [north, east]], padding=(18, 18))
    Fullscreen().add_to(m)
    return m


def active_fires_thumbnail(aoi: gpd.GeoDataFrame, fires_aoi: gpd.GeoDataFrame | None, size: int = 420) -> bytes:
    aoi = aoi.to_crs(4326)
    geom = aoi.geometry.iloc[0]
    west, south, east, north = geom.bounds
    dx = max(east - west, 0.01)
    dy = max(north - south, 0.01)
    pad_x = dx * 0.2 + 0.01
    pad_y = dy * 0.2 + 0.01
    west, south, east, north = west - pad_x, south - pad_y, east + pad_x, north + pad_y

    img = Image.new("RGBA", (size, size), (247, 247, 247, 255))
    draw = ImageDraw.Draw(img)

    def px(x):
        return int((x - west) / max(1e-9, east - west) * (size - 1))

    def py(y):
        return int((north - y) / max(1e-9, north - south) * (size - 1))

    draw.rectangle([0, 0, size - 1, size - 1], outline=(180, 180, 180, 255), width=1)
    a_w, a_s, a_e, a_n = geom.bounds
    pts = [(px(a_w), py(a_s)), (px(a_e), py(a_s)), (px(a_e), py(a_n)), (px(a_w), py(a_n)), (px(a_w), py(a_s))]
    draw.line(pts, fill=(0, 0, 0, 255), width=3)

    if isinstance(fires_aoi, gpd.GeoDataFrame) and not fires_aoi.empty:
        for _, row in fires_aoi.iterrows():
            x, y = px(row.geometry.x), py(row.geometry.y)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 90, 54, 220), outline=(140, 32, 16, 255), width=1)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


st.markdown(f"<p class='map-help'>{t('Desenați un poligon sau un dreptunghi. Harta interactivă a fost simplificată pentru a rămâne rapidă, iar miniaturile tematice sunt generate după analiză.', 'Draw a polygon or rectangle. The interactive map was simplified to stay fast, and thematic thumbnails are generated after analysis.')}</p>", unsafe_allow_html=True)
out = st_folium(build_map(), height=820, width=None, returned_objects=["all_drawings", "last_active_drawing"])

last = out.get("last_active_drawing") if out else None
if not last or not last.get("geometry"):
    st.warning(t("Desenați un poligon sau dreptunghi pe hartă pentru a activa analizorul AOI.", "Draw a polygon or rectangle on the map to activate the AOI analyser."))
    st.stop()

try:
    aoi = geojson_to_gdf(last["geometry"])
except Exception as exc:
    st.error(str(exc))
    st.stop()

aoi_area = area_ha(aoi)
lon, lat = centroid_lonlat(aoi)
bbox = bounds_csv(aoi)
geojson_str = json.dumps(last["geometry"], sort_keys=True)
aoi_signature = geojson_str

st.divider()
st.header(t("Analizor AOI", "AOI analyser"))

if aoi_area > MAX_AOI_HA:
    st.error(t(
        f"AOI-ul are {aoi_area:,.1f} ha și depășește limita admisă de {MAX_AOI_HA:,.0f} ha (50 km²). Redesenati un poligon mai mic pentru a evita blocarea aplicației.",
        f"The AOI is {aoi_area:,.1f} ha and exceeds the allowed limit of {MAX_AOI_HA:,.0f} ha (50 km²). Please redraw a smaller polygon to avoid app crashes.",
    ))
    st.stop()

run = st.button(t("▶ Analizează poligonul selectat", "▶ Analyse selected polygon"), type="primary", use_container_width=True)

previous_signature = st.session_state.get("analysis_aoi_signature")
if previous_signature != aoi_signature and not run:
    st.caption(t(
        f"AOI selectat: {aoi_area:,.1f} ha din maximum {MAX_AOI_HA:,.0f} ha. Apăsați **Analizează poligonul selectat** pentru interogarea seturilor de date online și generarea miniaturilor tematice.",
        f"Selected AOI: {aoi_area:,.1f} ha out of the maximum {MAX_AOI_HA:,.0f} ha. Click **Analyse selected polygon** to query the online datasets and generate thematic thumbnails.",
    ))
    st.stop()

if run:
    for _k in ["spread_env", "spread_sim", "spread_baseline", "spread_signature", "spread_seed", "spread_control_lines", "spread_manual_seed"]:
        st.session_state.pop(_k, None)
    result = {
        "project": {
            "name": "PREPARE – Proactive Resilience and Emergency Preparedness for Adaptive Response and Efficiency",
            "work_package": "WP4 – Forest fires / Incendii de pădure",
            "project_code": "PN-IV-P6-6.1-CoEx-2024-0102",
            "period": "2026–2030",
            "url": PROJECT_URL,
        },
        "aoi": {"area_ha": aoi_area, "centroid_lat": lat, "centroid_lon": lon},
        "active_fire": {},
        "burned_area": {},
        "weather_soil": {},
        "forest_biomass": {},
        "hazard": {},
        "effis_fire_danger": {
            "forecast_date": fwi_day.isoformat(),
            "layers": {k: (EFFIS_FORECAST_INFO[k]["ro"] if RO else EFFIS_FORECAST_INFO[k]["en"]) for k in EFFIS_FORECAST_LAYERS}
        },
        "warnings": {},
    }

    thumbs: dict[str, bytes] = {}

    with st.spinner(t(
        "Interogare meteo, umiditatea solului, FIRMS, EFFIS, ESA CCI Biomass, ESA WorldCover și generare miniaturi…",
        "Querying weather, soil moisture, FIRMS, EFFIS, ESA CCI Biomass, ESA WorldCover and generating thumbnails…",
    )):
        try:
            current, hourly = fetch_weather(lat, lon, forecast_days=3)
            result["weather_soil"] = current
            st.session_state["hourly"] = hourly
        except Exception as exc:
            current, hourly = {}, pd.DataFrame()
            result["warnings"]["weather"] = str(exc)

        try:
            fires = cached_firms(source, bbox, day_range)
            fires_aoi = filter_to_aoi(fires, aoi)
            result["active_fire"] = summarize_fires(fires_aoi)
            st.session_state["fires_aoi"] = fires_aoi
            thumbs[t("Focare active FIRMS", "FIRMS active fires")] = active_fires_thumbnail(aoi, fires_aoi)
        except Exception as exc:
            result["warnings"]["firms"] = str(exc)

        try:
            burned = cached_effis_burned(geojson_str)
            result["burned_area"] = summarize_burned_areas(burned, aoi)
            st.session_state["burned"] = burned
        except Exception as exc:
            result["warnings"]["effis_burned_area"] = str(exc)

        try:
            wc = cached_worldcover(geojson_str, aoi_area)
            result["forest_biomass"].update(wc)
            thumbs["ESA WorldCover 2021"] = cached_worldcover_thumb(geojson_str)
            if wc.get("worldcover_tile_warnings"):
                result["warnings"]["worldcover_partial"] = "; ".join(wc["worldcover_tile_warnings"])
        except Exception as exc:
            result["warnings"]["worldcover"] = str(exc)

        try:
            bio = cached_biomass(geojson_str, aoi_area)
            result["forest_biomass"].update(bio)
            thumbs[f"ESA CCI Biomass {CCI_YEAR}"] = cached_biomass_thumb(geojson_str)
            if bio.get("biomass_tile_warnings"):
                result["warnings"]["biomass_partial"] = "; ".join(bio["biomass_tile_warnings"])
        except Exception as exc:
            result["warnings"]["cci_biomass"] = str(exc)

        # EFFIS/GWIS thumbnails are downloaded server-side as real PNG bytes.
        # Requests use regional context plus viewer-style TIME intervals, then
        # the AOI outline is drawn locally. This avoids broken browser <img> cards.
        effis_products, effis_errors = cached_effis_gallery(geojson_str, fwi_day.isoformat())
        for product_key, message in effis_errors.items():
            result["warnings"][f"effis_{product_key.lower()}"] = message
        result["effis_fire_danger"]["available_products"] = {
            key: {
                "layer": value.get("layer"),
                "time": value.get("time"),
                "danger_class_dominant_aoi": value.get("danger_class"),
                "danger_class_max_aoi": value.get("danger_class_max"),
                "dominant_class_share_pct": value.get("danger_class_share_pct"),
            }
            for key, value in effis_products.items()
        }
        st.session_state["effis_products"] = effis_products

        if current:
            score, label, components = portal_hazard_score(current, ndmi=None)
            result["hazard"] = {
                "portal_screening_score_0_100": score,
                "screening_class": label,
                "components": components,
                "official_fwi_date": fwi_day.isoformat(),
                "official_fwi_note": "Use the generated EFFIS thumbnail maps for the official harmonized European fire-danger context.",
            }
        else:
            result["hazard"] = {
                "portal_screening_score_0_100": None,
                "screening_class": "Unavailable",
                "components": {},
                "official_fwi_date": fwi_day.isoformat(),
            }

    st.session_state["analysis"] = result
    st.session_state["analysis_aoi_signature"] = aoi_signature
    st.session_state["thumbnails"] = thumbs

result = st.session_state.get("analysis")
if not result:
    st.stop()

fire = result.get("active_fire", {})
burn = result.get("burned_area", {})
weather = result.get("weather_soil", {})
fb = result.get("forest_biomass", {})
haz = result.get("hazard", {})

hazard_labels_ro = {"Low": "Scăzut", "Moderate": "Moderat", "High": "Ridicat", "Very high": "Foarte ridicat", "Extreme": "Extrem", "Unavailable": "Indisponibil"}
hazard_label = hazard_labels_ro.get(haz.get("screening_class"), haz.get("screening_class")) if RO else haz.get("screening_class")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("AOI", f"{result['aoi']['area_ha']:,.1f} ha")
m2.metric(t("Acoperire forestieră", "Forest cover"), f"{fb.get('forest_fraction_pct', float('nan')):.1f}%" if fb.get("forest_fraction_pct") is not None else "n/a")
m3.metric(t("Biomasă medie AGB", "Mean AGB"), f"{fb.get('agb_mean_mg_ha', float('nan')):.1f} Mg/ha" if fb.get("agb_mean_mg_ha") is not None else "n/a")
m4.metric(t("Focare recente", "Recent hotspots"), fire.get("hotspots", "n/a"))
m5.metric(t("Suprafață arsă", "Burned area"), f"{burn.get('burned_area_ha', 0):,.1f} ha")
m6.metric(t("Screening hazard", "Hazard screening"), f"{haz.get('portal_screening_score_0_100','n/a')} / 100", hazard_label)

left, right = st.columns([1.05, 1.15], gap="large")
with left:
    st.subheader(t("Context meteo și umiditatea solului", "Weather and soil-moisture context"))
    weather_rows = [
        [t("Temperatură", "Temperature"), weather.get("temperature_2m"), "°C"],
        [t("Umiditate relativă", "Relative humidity"), weather.get("relative_humidity_2m"), "%"],
        [t("Viteza vântului", "Wind speed"), weather.get("wind_speed_10m"), "km/h"],
        [t("Direcția vântului (din)", "Wind direction (from)"), weather.get("wind_direction_10m"), "°"],
        [t("Rafala de vânt", "Wind gust"), weather.get("wind_gusts_10m"), "km/h"],
        [t("Deficit de presiune a vaporilor", "Vapour pressure deficit"), weather.get("vapour_pressure_deficit"), "kPa"],
        [t("Precipitații", "Precipitation"), weather.get("precipitation"), "mm"],
        [t("Umiditatea solului 0–1 cm", "Soil moisture 0–1 cm"), weather.get("soil_moisture_0_to_1cm"), "m³/m³"],
        [t("Umiditatea solului 1–3 cm", "Soil moisture 1–3 cm"), weather.get("soil_moisture_1_to_3cm"), "m³/m³"],
        [t("Umiditatea solului 3–9 cm", "Soil moisture 3–9 cm"), weather.get("soil_moisture_3_to_9cm"), "m³/m³"],
        [t("Umiditatea solului 9–27 cm", "Soil moisture 9–27 cm"), weather.get("soil_moisture_9_to_27cm"), "m³/m³"],
        [t("Umiditatea solului 27–81 cm", "Soil moisture 27–81 cm"), weather.get("soil_moisture_27_to_81cm"), "m³/m³"],
    ]
    st.dataframe(pd.DataFrame(weather_rows, columns=[t("Indicator", "Indicator"), t("Valoare", "Value"), t("Unitate", "Unit")]), hide_index=True, use_container_width=True)

with right:
    st.subheader(t("Acoperirea forestieră și biomasa", "Forest cover and biomass"))
    forest_rows = [
        [t("Suprafață cu arbori (WorldCover)", "Tree-cover area (WorldCover)"), fb.get("forest_area_ha"), "ha"],
        [t("Fracție cu acoperire arborată", "Tree-cover fraction"), fb.get("forest_fraction_pct"), "%"],
        [t("Biomasă supraterană medie ESA CCI", "Mean ESA CCI above-ground biomass density"), fb.get("agb_mean_mg_ha"), "Mg/ha"],
        [t("Biomasă mediană ESA CCI", "Median ESA CCI AGB"), fb.get("agb_median_mg_ha"), "Mg/ha"],
        [t("Percentila 90 ESA CCI AGB", "90th percentile ESA CCI AGB"), fb.get("agb_p90_mg_ha"), "Mg/ha"],
        [t("AGB medie pentru pixelii cu biomasă > 0", "Mean AGB for biomass-positive pixels"), fb.get("agb_mean_positive_mg_ha"), "Mg/ha"],
        [t("Incertitudine medie AGB (deviație standard)", "Mean AGB uncertainty (standard deviation)"), fb.get("agb_standard_deviation_mean_mg_ha"), "Mg/ha"],
        [t("Pixeli CCI analizați", "CCI pixels sampled"), fb.get("agb_pixels"), "100-m cells"],
        [t("Biomasă supraterană totală estimată în AOI", "Estimated total AGB over AOI"), fb.get("agb_total_aoi_mg_est"), "Mg"],
    ]
    st.dataframe(pd.DataFrame(forest_rows, columns=[t("Indicator", "Indicator"), t("Valoare", "Value"), t("Unitate", "Unit")]), hide_index=True, use_container_width=True)
    st.caption(t(
        f"ESA CCI Biomass v{CCI_VERSION} pentru {CCI_YEAR} are rezoluție de 100 m. AGB este masa uscată a componentelor lemnoase supraterane pe hectar. Valorile din portal sunt estimări de teledetecție pentru screening, nu inventar forestier de teren.",
        f"ESA CCI Biomass v{CCI_VERSION} for {CCI_YEAR} has 100 m grid spacing. AGB is oven-dry above-ground woody biomass per hectare. Portal values are remote-sensing screening estimates, not a field forest inventory.",
    ))

st.subheader(t("Context EFFIS pentru pericolul de incendiu", "EFFIS fire-danger context"))
effis_rows = []
for short_name in ["FUEL", "MODIS_HS", "FWI", "FFMC", "DC", "ISI", "BUI", "NFDRS_IC", "NFDRS_ROS"]:
    effis_rows.append({
        t("Indice", "Index"): short_name,
        t("Descriere", "Description"): EFFIS_FORECAST_INFO[short_name]["ro"] if RO else EFFIS_FORECAST_INFO[short_name]["en"],
        t("Data", "Date"): (
            "2017 / static" if short_name == "FUEL"
            else t("sezon curent", "current season") if short_name == "MODIS_HS"
            else result.get("effis_fire_danger", {}).get("forecast_date")
        ),
    })
st.dataframe(pd.DataFrame(effis_rows), hide_index=True, use_container_width=True)
st.caption(t(
    "Pentru miniaturi folosim exact serviciul operațional GWIS și denumirile de strat publicate: ecmwf.fwi, ecmwf.ffmc, ecmwf.dc, ecmwf.isi, ecmwf.bui, ecmwf.nfdrs.ic și ecmwf.nfdrs.ros. Data este transmisă ca TIME=YYYY-MM-DD.",
    "For thumbnails we use the operational GWIS service and the published layer IDs exactly: ecmwf.fwi, ecmwf.ffmc, ecmwf.dc, ecmwf.isi, ecmwf.bui, ecmwf.nfdrs.ic and ecmwf.nfdrs.ros. The date is passed as TIME=YYYY-MM-DD.",
))
with st.expander(t("Pragurile oficiale EFFIS pentru clasele de pericol", "Official EFFIS fire-danger class thresholds")):
    thresholds_df = pd.DataFrame(EFFIS_CLASS_THRESHOLDS)
    if RO:
        thresholds_df = thresholds_df.rename(columns={"Class": "Clasă"})
        thresholds_df["Clasă"] = thresholds_df["Clasă"].replace({
            "Low": "Scăzut", "Moderate": "Moderat", "High": "Ridicat",
            "Very High": "Foarte ridicat", "Extreme": "Extrem", "Very Extreme": "Foarte extrem"
        })
    st.dataframe(thresholds_df, hide_index=True, use_container_width=True)
st.caption(t(
    "Pentru FWI, FFMC, DC, ISI și BUI, denumirile claselor urmează scala EFFIS cu 6 clase. Pentru NFDRS IC și ROS, portalul citește clasa direct din culorile hărții GWIS din interiorul AOI; nu aplică praguri numerice Canadian-FWI acestor indici NFDRS.",
    "For FWI, FFMC, DC, ISI and BUI, class names follow the six-class EFFIS scale. For NFDRS IC and ROS, the portal reads the class directly from GWIS map colours inside the AOI; it does not apply Canadian-FWI numeric thresholds to those NFDRS indices.",
))

thumbs = st.session_state.get("thumbnails", {})
effis_products = st.session_state.get("effis_products", {})

def _data_uri(img_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(img_bytes).decode("ascii")

def _thumb_card(src: str, title: str, meta: str, desc: str, badge: str, danger_html: str = "") -> str:
    return f"""
    <div class="thumb-card">
      <img src="{src}" alt="{title}">
      <div class="thumb-body">
        <div class="thumb-badge">{badge}</div>
        <div class="thumb-title">{title}</div>
        <div class="thumb-meta">{meta}</div>
        <div class="thumb-desc">{desc}</div>
        {danger_html}
      </div>
    </div>
    """

_DANGER_RO = {
    "Low": "Scăzut",
    "Moderate": "Moderat",
    "High": "Ridicat",
    "Very High": "Foarte ridicat",
    "Extreme": "Extrem",
    "Very Extreme": "Foarte extrem",
}

def _danger_class_html(product: dict) -> str:
    dominant = product.get("danger_class")
    maximum = product.get("danger_class_max")
    share = product.get("danger_class_share_pct")
    if not dominant:
        return f'<div class="danger-result"><b>{t("Clasa de pericol în AOI", "Fire-danger class in AOI")}:</b> {t("indisponibilă", "unavailable")}</div>'
    dom_label = _DANGER_RO.get(dominant, dominant) if RO else dominant
    max_label = _DANGER_RO.get(maximum, maximum) if RO else maximum
    share_text = f" ({share:.0f}% {t('din AOI', 'of AOI')})" if share is not None else ""
    if maximum and maximum != dominant:
        detail = t(
            f"Clasă dominantă: <b>{dom_label}</b>{share_text} · maxim intersectat: <b>{max_label}</b>",
            f"Dominant class: <b>{dom_label}</b>{share_text} · maximum intersecting class: <b>{max_label}</b>",
        )
    else:
        detail = t(
            f"Clasa de pericol în AOI: <b>{dom_label}</b>{share_text}",
            f"Fire-danger class in AOI: <b>{dom_label}</b>{share_text}",
        )
    return f'<div class="danger-result">🔥 {detail}</div>'

if thumbs or effis_products:
    st.subheader(t("Galerie tematică pentru AOI", "AOI thematic gallery"))
    st.caption(t(
        "Harta principală rămâne simplă. Produsele tematice sunt afișate aici ca imagini independente, ceea ce reduce încărcarea hărții și permite compararea rapidă a combustibilului, vegetației și pericolului meteo.",
        "The main map stays simple. Thematic products are shown here as independent images, reducing map load and making fuel, vegetation and fire-weather context easy to compare.",
    ))

    main_cards = []
    if t("Focare active FIRMS", "FIRMS active fires") in thumbs:
        main_cards.append((
            _data_uri(thumbs[t("Focare active FIRMS", "FIRMS active fires")]),
            t("Focare active FIRMS", "FIRMS active fires"),
            f"{source} • {day_range} {t('zile', 'days')}",
            t("Anomalii termice detectate în interiorul AOI; intensitatea este descrisă separat prin FRP.", "Thermal anomalies detected inside the AOI; intensity is described separately by FRP."),
            "NASA FIRMS",
        ))
    if "ESA WorldCover 2021" in thumbs:
        main_cards.append((
            _data_uri(thumbs["ESA WorldCover 2021"]),
            "ESA WorldCover 2021",
            "10 m • 2021",
            t("Contextul acoperirii terenului și proporția de acoperire arborată în AOI.", "Land-cover context and tree-cover proportion within the AOI."),
            "ESA",
        ))
    if f"ESA CCI Biomass {CCI_YEAR}" in thumbs:
        main_cards.append((
            _data_uri(thumbs[f"ESA CCI Biomass {CCI_YEAR}"]),
            f"ESA CCI Biomass {CCI_YEAR}",
            f"100 m • AGB Mg/ha • v{CCI_VERSION}",
            t("Distribuția biomasei supraterane, utilă ca indicator de încărcare și structură a combustibilului.", "Above-ground biomass distribution, useful as a fuel-load and structural context indicator."),
            "ESA CCI",
        ))

    if main_cards:
        st.markdown(f"### {t('Combustibil, vegetație și focare', 'Fuel, vegetation and active fires')}")
        for i in range(0, len(main_cards), 3):
            cols = st.columns(3)
            for col, card in zip(cols, main_cards[i:i+3]):
                with col:
                    st.markdown(_thumb_card(*card), unsafe_allow_html=True)

    if effis_products:
        st.markdown(f"### {t('Produse Copernicus EFFIS / GWIS', 'Copernicus EFFIS / GWIS products')}")
        st.caption(t(
            "Harta de combustibil este citită din /effis, iar hotspot-urile și indicii meteo sunt citite din /gwis. Pentru indicii GWIS ECMWF folosim exact TIME=YYYY-MM-DD și exact ID-urile de strat furnizate de serviciul operațional.",
            "The fuel map is read from /effis, while hotspots and fire-weather products are read from /gwis. For GWIS ECMWF products we use the exact TIME=YYYY-MM-DD pattern and exact operational layer IDs.",
        ))

        context_order = ["FUEL", "MODIS_HS"]
        context_cards = []
        for product_key in context_order:
            if product_key not in effis_products:
                continue
            product = effis_products[product_key]
            title = "EFFIS European Fuel Map" if product_key == "FUEL" else "GWIS MODIS hotspots"
            badge = "Copernicus EFFIS" if product_key == "FUEL" else "Copernicus GWIS"
            meta = (
                "European Fuel Map 2017 • fuel_map"
                if product_key == "FUEL"
                else f"{product.get('time')} • modis.hs"
            )
            desc = EFFIS_FORECAST_INFO[product_key]["ro"] if RO else EFFIS_FORECAST_INFO[product_key]["en"]
            context_cards.append((_data_uri(product["png"]), title, meta, desc, badge))

        if context_cards:
            st.markdown(f"#### {t('Combustibil și observații satelitare EFFIS/GWIS', 'EFFIS/GWIS fuel and satellite observations')}")
            cols = st.columns(2)
            for col, card in zip(cols, context_cards):
                with col:
                    st.markdown(_thumb_card(*card), unsafe_allow_html=True)

        st.markdown(f"#### {t('Pericol meteo EFFIS', 'EFFIS fire-weather danger')}")
        effis_order = ["FWI", "FFMC", "DC", "ISI", "BUI", "NFDRS_IC", "NFDRS_ROS"]
        for i in range(0, len(effis_order), 4):
            cols = st.columns(4)
            for col, short_name in zip(cols, effis_order[i:i+4]):
                with col:
                    if short_name not in effis_products:
                        st.warning(t(f"EFFIS {short_name}: imagine indisponibilă pentru data selectată.", f"EFFIS {short_name}: image unavailable for the selected date."))
                        continue
                    product = effis_products[short_name]
                    desc = EFFIS_FORECAST_INFO[short_name]["ro"] if RO else EFFIS_FORECAST_INFO[short_name]["en"]
                    meta = f"{product.get('time')} • GWIS • {product.get('layer')}"
                    st.markdown(
                        _thumb_card(
                            _data_uri(product["png"]),
                            f"EFFIS {short_name}",
                            meta,
                            desc,
                            "Copernicus GWIS",
                            _danger_class_html(product),
                        ),
                        unsafe_allow_html=True,
                    )

classes = fb.get("worldcover_classes") or {}
if classes:
    st.subheader(t("Compoziția ESA WorldCover", "ESA WorldCover composition"))
    wc_ro = {10:"Acoperire arborată",20:"Tufăriș",30:"Pajiști",40:"Teren agricol",50:"Suprafață construită",60:"Vegetație rară / sol neacoperit",70:"Zăpadă și gheață",80:"Corpuri de apă permanente",90:"Zonă umedă erbacee",95:"Mangrove",100:"Mușchi și licheni"}
    class_rows = []
    for name, d in classes.items():
        code = d.get("class_code")
        class_rows.append({
            t("Clasa de acoperire", "Land-cover class"): wc_ro.get(code, name) if RO else name,
            t("Cod", "Code"): code,
            t("Suprafață estimată (ha)", "Area (ha, estimated)"): d.get("area_ha_est"),
            "AOI (%)": d.get("percent"),
        })
    df_classes = pd.DataFrame(class_rows)
    st.dataframe(df_classes.sort_values("AOI (%)", ascending=False), hide_index=True, use_container_width=True)

hourly = st.session_state.get("hourly", pd.DataFrame())
if not hourly.empty:
    st.subheader(t("Următoarele 72 de ore", "Next 72 hours"))
    chart_df = hourly[[c for c in ["time", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "soil_moisture_0_to_1cm"] if c in hourly.columns]].copy()
    melted = chart_df.melt(id_vars="time", var_name="indicator", value_name="value")
    fig = px.line(melted, x="time", y="value", color="indicator", facet_row="indicator", height=620)
    fig.update_yaxes(matches=None)
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader(t("Detecții de incendiu în poligonul selectat", "Fire detections in selected polygon"))
fires_aoi = st.session_state.get("fires_aoi")
if isinstance(fires_aoi, gpd.GeoDataFrame) and not fires_aoi.empty:
    cols = [c for c in ["acq_date", "acq_time", "satellite", "instrument", "confidence", "frp", "bright_ti4", "latitude", "longitude"] if c in fires_aoi.columns]
    st.dataframe(fires_aoi[cols], hide_index=True, use_container_width=True)
else:
    st.caption(t("Nu au fost returnate detecții FIRMS pentru acest AOI și interval.", "No FIRMS detections were returned for this AOI/window."))

st.divider()
st.header(t("🔥 Simulator de propagare a incendiului", "🔥 Fire Spread Scenario Simulator"))
st.markdown(
    t(
        "<div class='spread-box'><b>Etapa 2 după evaluarea situației actuale.</b> Modelul folosește WorldCover, biomasa ESA CCI, panta din Copernicus DEM și condițiile meteo curente ca punct de pornire. Puteți modifica vântul și umiditatea combustibilului și puteți desena linii ipotetice de control pentru a compara scenarii.</div>",
        "<div class='spread-box'><b>Stage 2 after reviewing the current situation.</b> The model uses WorldCover, ESA CCI biomass, Copernicus DEM slope and current weather as its starting point. You can change wind and fuel moisture and draw hypothetical control lines to compare scenarios.</div>",
    ),
    unsafe_allow_html=True,
)
st.markdown(
    t(
        "<div class='spread-warning'><b>Model experimental de screening.</b> Nu este un model operațional validat pentru dispecerizare, zone de siguranță sau rute de evacuare. Poziționarea echipelor trebuie decisă de comanda incidentului pe baza observațiilor din teren, accesului, barierelor și comportamentului real al focului.</div>",
        "<div class='spread-warning'><b>Experimental screening model.</b> It is not a validated operational model for dispatch, safety zones or escape routes. Crew positioning must be decided by incident command using field observations, access, barriers and actual fire behaviour.</div>",
    ),
    unsafe_allow_html=True,
)

with st.expander(t("Cum funcționează modelul de propagare", "How the spread model works")):
    st.markdown(t(
        """
Modelul construiește o grilă adaptivă de **50–100 m** în AOI și calculează un timp de sosire la fiecare celulă. Este un model direcțional de tip *least-cost / travel-time*, inspirat de variabilele utilizate în modelele clasice de propagare a focului, dar **nu este o implementare calibrată Rothermel/BehavePlus**.

- **ESA WorldCover** stabilește tipul relativ de combustibil;
- **ESA CCI Biomass** modifică moderat încărcarea combustibilului;
- **Copernicus DEM GLO-30** introduce efectul direcțional al pantei;
- **vântul** accelerează propagarea în direcția *downwind* și o încetinește contra vântului;
- **umiditatea combustibilului fin** reduce sau crește viteza scenariului;
- o **linie albastră de control** desenată de utilizator poate fi tratată ca zonă de încetinire/barieră ipotetică.

Culorile hărții rezultat reprezintă **timpul modelat de sosire**, nu intensitatea flăcării.
""",
        """
The model builds an adaptive **50–100 m grid** over the AOI and calculates an arrival time for each cell. It is a directional *least-cost / travel-time* model inspired by variables used in classical fire-spread models, but it is **not a calibrated Rothermel/BehavePlus implementation**.

- **ESA WorldCover** defines relative fuel type;
- **ESA CCI Biomass** moderately adjusts fuel loading;
- **Copernicus DEM GLO-30** introduces directional slope effects;
- **wind** accelerates spread downwind and slows backing spread;
- **fine-fuel moisture** changes scenario spread speed;
- a user-drawn **blue control line** can be treated as a hypothetical slowing/barrier zone.

Result-map colours represent **modelled arrival time**, not flame intensity.
""",
    ))
    fuel_rows = []
    for code in [10, 20, 30, 40, 60, 90]:
        fuel_rows.append({
            t("WorldCover", "WorldCover"): FUEL_NAMES.get(code, str(code)),
            t("Cod", "Code"): code,
            t("Coeficient ROS de bază", "Baseline ROS coefficient"): BASE_ROS_M_MIN.get(code),
            t("Unitate internă", "Internal unit"): "m/min",
        })
    st.dataframe(pd.DataFrame(fuel_rows), hide_index=True, use_container_width=True)
    st.caption(t(
        "Valorile ROS de bază sunt coeficienți interni pentru comparația scenariilor și trebuie calibrați cu observații locale înainte de orice utilizare operațională.",
        "Baseline ROS values are internal scenario-comparison coefficients and require calibration against local observations before any operational use.",
    ))

fires_for_seed = st.session_state.get("fires_aoi")
firms_seed = _latest_firms_seed(fires_for_seed)
centroid_seed = (lon, lat)
seed_options = []
if firms_seed:
    seed_options.append(t("Ultimul hotspot FIRMS din AOI", "Most recent FIRMS hotspot in AOI"))
seed_options.extend([
    t("Centroidul AOI", "AOI centroid"),
    t("Alege prin click pe hartă", "Choose by clicking on the map"),
])
seed_mode = st.radio(t("Punct de pornire al simulării", "Simulation start point"), seed_options, horizontal=True)

if firms_seed and seed_mode == seed_options[0]:
    base_seed = firms_seed
elif seed_mode == t("Centroidul AOI", "AOI centroid"):
    base_seed = centroid_seed
else:
    base_seed = st.session_state.get("spread_manual_seed", centroid_seed)

# Finer grids for smaller AOIs make the simulated front less blocky while
# keeping the 5,000-ha maximum computationally manageable.
spread_grid_m = 50 if aoi_area <= 1500 else (75 if aoi_area <= 3500 else 100)
st.caption(t(
    f"Rezoluția grilei pentru acest AOI: **{spread_grid_m} m**.",
    f"Spread-grid resolution for this AOI: **{spread_grid_m} m**.",
))

setup_left, setup_right = st.columns([1.55, 1.0], gap="large")
with setup_right:
    current_wind = float(weather.get("wind_speed_10m") or 10.0)
    current_dir = float(weather.get("wind_direction_10m") or 270.0)
    default_fm = estimate_fine_fuel_moisture_proxy(weather)
    st.markdown(f"**{t('Condiții scenariu', 'Scenario conditions')}**")
    horizon_h = st.slider(t("Orizont de simulare (ore)", "Simulation horizon (hours)"), 0.5, 24.0, 6.0, 0.5)
    wind_speed_s = st.slider(t("Viteza vântului la 10 m (km/h)", "10-m wind speed (km/h)"), 0.0, 80.0, float(np.clip(round(current_wind, 1), 0, 80)), 0.5)
    wind_dir_s = st.slider(t("Direcția DIN CARE bate vântul (°)", "Wind direction FROM (°)"), 0, 359, int(round(current_dir)) % 360, 1)
    fuel_moisture_s = st.slider(t("Umiditatea combustibilului fin – scenariu (%)", "Fine-fuel moisture – scenario (%)"), 5.0, 40.0, float(round(default_fm, 1)), 0.5)
    control_width_s = st.slider(t("Lățimea liniei ipotetice de control (m)", "Hypothetical control-line width (m)"), 20, 200, 60, 10)
    control_effect_s = st.slider(t("Eficiența ipotetică a liniei (%)", "Hypothetical line effectiveness (%)"), 0, 100, 90, 5)
    if st.button(t("Șterge liniile de control", "Clear control lines"), use_container_width=True):
        st.session_state["spread_control_lines"] = []
        st.rerun()
    st.caption(t(
        f"Condiții curente la analiza AOI: vânt {current_wind:.1f} km/h din {current_dir:.0f}°. Valoarea inițială pentru umiditatea combustibilului este doar un proxy meteo ({default_fm:.1f}%), nu o măsurare FFMC/fuel-stick.",
        f"Current AOI conditions: wind {current_wind:.1f} km/h from {current_dir:.0f}°. The initial fuel-moisture value is only a weather-based proxy ({default_fm:.1f}%), not an FFMC/fuel-stick measurement.",
    ))

with setup_left:
    spread_setup_out = st_folium(
        _build_spread_setup_map(aoi, base_seed[0], base_seed[1], st.session_state.get("spread_control_lines", [])),
        height=520,
        width=None,
        key=f"spread_setup_map_{hashlib.md5(aoi_signature.encode()).hexdigest()[:10]}",
        returned_objects=["last_clicked", "all_drawings"],
    )
    st.caption(t(
        "Faceți click pentru a muta seed-ul când este selectat modul manual. Instrumentul linie albastră permite desenarea uneia sau mai multor linii ipotetice de control; acestea nu reprezintă recomandări de amplasare a personalului.",
        "Click to move the seed when manual mode is selected. The blue-line tool lets you draw one or more hypothetical control lines; these are not crew-positioning recommendations.",
    ))

clicked = (spread_setup_out or {}).get("last_clicked")
if seed_mode == t("Alege prin click pe hartă", "Choose by clicking on the map") and clicked:
    click_lon, click_lat = float(clicked["lng"]), float(clicked["lat"])
    if _point_inside_aoi(aoi, click_lon, click_lat):
        st.session_state["spread_manual_seed"] = (click_lon, click_lat)
        base_seed = (click_lon, click_lat)
    else:
        st.warning(t("Click-ul pentru seed este în afara AOI; se păstrează ultimul seed valid.", "The seed click is outside the AOI; the last valid seed is kept."))
        base_seed = st.session_state.get("spread_manual_seed", centroid_seed)

current_lines = _extract_control_lines((spread_setup_out or {}).get("all_drawings"))
if current_lines:
    st.session_state["spread_control_lines"] = current_lines
control_lines = st.session_state.get("spread_control_lines", [])

scenario_signature = json.dumps({
    "aoi": aoi_signature,
    "seed": [round(base_seed[0], 6), round(base_seed[1], 6)],
    "horizon_h": horizon_h,
    "grid_m": spread_grid_m,
    "wind": wind_speed_s,
    "wind_dir": wind_dir_s,
    "fm": fuel_moisture_s,
    "control_width": control_width_s,
    "control_effect": control_effect_s,
    "lines": control_lines,
}, sort_keys=True, default=str)

run_spread = st.button(t("🔥 Rulează scenariul de propagare", "🔥 Run fire-spread scenario"), type="primary", use_container_width=True)
if run_spread:
    with st.spinner(t(
        f"Construire grilă {spread_grid_m} m: WorldCover + ESA CCI Biomass + Copernicus DEM și calculul timpului de sosire…",
        f"Building {spread_grid_m}-m grid: WorldCover + ESA CCI Biomass + Copernicus DEM and calculating arrival time…",
    )):
        try:
            spread_env = cached_spread_environment(geojson_str, spread_grid_m)
            spread_sim = simulate_spread(
                spread_env,
                ignition_lon=base_seed[0],
                ignition_lat=base_seed[1],
                horizon_minutes=int(round(horizon_h * 60)),
                wind_speed_kmh=wind_speed_s,
                wind_direction_from_deg=wind_dir_s,
                fine_fuel_moisture_pct=fuel_moisture_s,
                control_lines_geojson=control_lines,
                control_line_width_m=control_width_s,
                control_effectiveness_pct=control_effect_s,
            )
            baseline_sim = None
            if control_lines and control_effect_s > 0:
                baseline_sim = simulate_spread(
                    spread_env,
                    ignition_lon=base_seed[0],
                    ignition_lat=base_seed[1],
                    horizon_minutes=int(round(horizon_h * 60)),
                    wind_speed_kmh=wind_speed_s,
                    wind_direction_from_deg=wind_dir_s,
                    fine_fuel_moisture_pct=fuel_moisture_s,
                    control_lines_geojson=None,
                    control_line_width_m=control_width_s,
                    control_effectiveness_pct=0,
                )
            st.session_state["spread_env"] = spread_env
            st.session_state["spread_sim"] = spread_sim
            st.session_state["spread_baseline"] = baseline_sim
            st.session_state["spread_signature"] = scenario_signature
            st.session_state["spread_seed"] = base_seed
            result["fire_spread_scenario"] = {
                "model": "PREPARE experimental directional travel-time screening model",
                "grid_m": spread_env["cell_size_m"],
                "horizon_hours": horizon_h,
                "wind_speed_kmh": wind_speed_s,
                "wind_direction_from_deg": wind_dir_s,
                "fine_fuel_moisture_proxy_pct": fuel_moisture_s,
                "predicted_area_ha_at_horizon": spread_sim["area_ha"],
                "reached_cells_at_horizon": spread_sim.get("reached_cells"),
                "maximum_head_distance_m": spread_sim["max_distance_m"],
                "head_bearing_deg": spread_sim["head_bearing_deg"],
                "maximum_modelled_ros_m_min": spread_sim["max_ros_m_min"],
                "reaches_aoi_boundary": spread_sim["reaches_aoi_boundary"],
                "control_lines_drawn": len(control_lines),
            }
            st.session_state["analysis"] = result
        except Exception as exc:
            st.error(t(f"Simulatorul nu a putut fi rulat: {exc}", f"The simulator could not run: {exc}"))

spread_sim = st.session_state.get("spread_sim")
spread_env = st.session_state.get("spread_env")
spread_signature = st.session_state.get("spread_signature")
if spread_sim is not None and spread_env is not None:
    if spread_signature != scenario_signature:
        st.info(t("Parametrii au fost modificați după ultima simulare. Apăsați din nou **Rulează scenariul de propagare** pentru actualizare.", "Parameters have changed since the last simulation. Click **Run fire-spread scenario** again to update."))
    spread_seed = st.session_state.get("spread_seed", base_seed)
    horizon_min = int(spread_sim["horizon_minutes"])
    frame_step = 15 if horizon_min <= 360 else (30 if horizon_min <= 720 else 60)
    baseline_sim = st.session_state.get("spread_baseline")
    reduction = None
    if baseline_sim and baseline_sim.get("area_ha", 0) > 0:
        reduction = 100.0 * (baseline_sim["area_ha"] - spread_sim["area_ha"]) / baseline_sim["area_ha"]

    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    sm1.metric(t("Arie la orizont", "Area at horizon"), f"{spread_sim['area_ha']:,.1f} ha")
    sm2.metric(t("Celule atinse", "Reached cells"), f"{spread_sim.get('reached_cells', 0):,}")
    sm3.metric(t("Distanță maximă", "Maximum distance"), f"{spread_sim['max_distance_m']/1000:.2f} km")
    sm4.metric(t("Direcția capului", "Head direction"), f"{spread_sim['head_bearing_deg']:.0f}° {bearing_label(spread_sim['head_bearing_deg'])}")
    sm5.metric(t("ROS maxim modelat", "Max modelled ROS"), f"{spread_sim['max_ros_m_min']:.1f} m/min")

    play_tab, inspect_tab = st.tabs([
        t("▶️ Play – propagarea în timp", "▶️ Play – spread through time"),
        t("🔎 Inspectează un moment", "🔎 Inspect a time"),
    ])

    with play_tab:
        st.caption(t(
            f"Apăsați ▶ în colțul din stânga-jos al hărții. Pasul animației este {frame_step} minute; 00:00 reprezintă momentul aprinderii. Celulele rămân vizibile după ce sunt atinse, astfel încât frontul este cumulativ.",
            f"Press ▶ in the lower-left map control. Animation step is {frame_step} minutes; 00:00 is ignition. Cells remain visible after arrival, so the footprint is cumulative.",
        ))
        anim_map = _build_spread_animation_map(
            aoi, spread_env, spread_sim, spread_seed[0], spread_seed[1], control_lines, frame_step
        )
        components.html(anim_map.get_root().render(), height=650, scrolling=False)

    with inspect_tab:
        default_display = min(horizon_min, 60)
        display_min = st.slider(
            t("Moment afișat pe hartă (minute de la seed)", "Map time (minutes from seed)"),
            frame_step,
            max(frame_step, horizon_min),
            max(frame_step, default_display),
            frame_step,
            key="spread_display_time",
        )
        arrival = spread_sim["arrival_min"]
        display_area = float((np.isfinite(arrival) & (arrival <= display_min)).sum() * (spread_env["cell_size_m"] ** 2) / 10_000.0)
        st.metric(t(f"Arie modelată la {display_min} min", f"Modelled area at {display_min} min"), f"{display_area:,.1f} ha")
        st_folium(
            _build_spread_result_map(aoi, spread_env, spread_sim, display_min, spread_seed[0], spread_seed[1], control_lines),
            height=620,
            width=None,
            key=f"spread_result_map_{display_min}",
            returned_objects=[],
        )

    st.markdown(f"**{t('Repere pentru planificarea scenariilor', 'Scenario-planning cues')}**")
    cue1, cue2 = st.columns(2, gap="large")
    with cue1:
        st.write(t(
            f"• Sectorul de cap al frontului este orientat aproximativ la **{spread_sim['head_bearing_deg']:.0f}° ({bearing_label(spread_sim['head_bearing_deg'])})** în acest scenariu.",
            f"• The head-fire sector is oriented approximately **{spread_sim['head_bearing_deg']:.0f}° ({bearing_label(spread_sim['head_bearing_deg'])})** in this scenario.",
        ))
        if spread_sim.get("reaches_aoi_boundary"):
            st.warning(t("Frontul modelat atinge limita AOI înainte de sfârșitul scenariului. Rezultatul nu descrie propagarea dincolo de această limită.", "The modelled front reaches the AOI boundary before the scenario ends. The result does not describe spread beyond that boundary."))
    with cue2:
        if reduction is not None:
            st.success(t(
                f"Linia/liniile ipotetice desenate reduc aria modelată la orizont cu **{max(reduction, 0):.1f}%** față de același scenariu fără linie.",
                f"The drawn hypothetical line(s) reduce modelled area at the horizon by **{max(reduction, 0):.1f}%** compared with the same scenario without a line.",
            ))
        else:
            st.info(t("Desenați o linie albastră în harta de configurare și rulați din nou pentru a testa efectul unei poziții ipotetice de control.", "Draw a blue line in the setup map and rerun to test the effect of a hypothetical control-line position."))
    st.caption(t(
        "Folosiți rezultatul pentru compararea alternativelor și pentru a identifica sectoare potențial critice, nu pentru stabilirea unei zone de siguranță sau a poziției exacte a pompierilor.",
        "Use the result to compare alternatives and identify potentially critical sectors, not to define a safety zone or exact firefighter position.",
    ))
    env_warnings = spread_env.get("warnings", {})
    if any(env_warnings.values()):
        with st.expander(t("Avertismente date simulator", "Simulator data warnings")):
            for key, vals in env_warnings.items():
                for v in vals:
                    st.warning(f"{key}: {v}")

if result.get("warnings"):
    with st.expander(t("Avertismente privind sursele de date / module indisponibile", "Data-source warnings / unavailable modules")):
        for k, v in result["warnings"].items():
            st.warning(f"{k}: {v}")

st.subheader(t("Descărcare raport și AOI", "Download report and AOI"))
html_report = make_html_report(result, language="ro" if RO else "en")
geojson_bytes = aoi.to_json().encode("utf-8")
json_bytes = json.dumps(result, ensure_ascii=False, indent=2, default=str).encode("utf-8")

b1, b2, b3 = st.columns(3)
b1.download_button(t("⬇ Raport HTML", "⬇ HTML report"), html_report.encode("utf-8"), "PREPARE_WP4_wildfire_AOI_report.html", "text/html", use_container_width=True)
b2.download_button(t("⬇ Analiză JSON", "⬇ Analysis JSON"), json_bytes, "PREPARE_WP4_wildfire_AOI_analysis.json", "application/json", use_container_width=True)
b3.download_button(t("⬇ AOI GeoJSON", "⬇ AOI GeoJSON"), geojson_bytes, "PREPARE_WP4_wildfire_AOI.geojson", "application/geo+json", use_container_width=True)

st.caption(t(
    "Notă operațională: detecțiile FIRMS sunt anomalii termice satelitare și pot include surse de căldură care nu sunt incendii de vegetație; umiditatea solului Open-Meteo este modelată; ESA CCI Biomass și WorldCover au incertitudini spațiale și temporale; miniaturile EFFIS sunt destinate suportului de cercetare și screening. Portalul este pentru cercetare, screening și conștientizare situațională, nu pentru comandă operativă în situații de urgență.",
    "Operational note: FIRMS detections are satellite thermal anomalies and can include non-wildfire heat sources; Open-Meteo soil moisture is modelled; ESA CCI Biomass and WorldCover have spatial/temporal uncertainty; EFFIS thumbnails are meant for research support and screening. This portal is for research, screening and situational awareness, not emergency command decisions.",
))
