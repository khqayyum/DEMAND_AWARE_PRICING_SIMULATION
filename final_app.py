import os
import re
import json
import math
import numpy as np
import pandas as pd
import streamlit as st

try:
    import geopandas as gpd
except Exception:
    gpd = None
import joblib
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ============================================================
# ENV + PAGE SETUP
# ============================================================
load_dotenv()

st.set_page_config(
    page_title="NYC Yellow Taxi Pricing Intelligence",
    page_icon="NYC",
    layout="wide"
)

st.markdown("""
<style>
    :root {
        --taxi-yellow:#FACC15;
        --taxi-yellow-light:#FFE580;
        --taxi-black:#0B0D12;
        --taxi-panel:#151821;
        --taxi-panel-2:#20232D;
        --taxi-text:#F4F4F5;
        --taxi-muted:#A1A1AA;
        --taxi-border:rgba(255,255,255,.12);
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(250,204,21,.20), transparent 28%),
            radial-gradient(circle at 86% 6%, rgba(250,204,21,.10), transparent 24%),
            linear-gradient(135deg,#05070A 0%,#0B0D12 48%,#111827 100%);
        color:var(--taxi-text);
    }

    .block-container {
        padding-top:1.35rem;
        padding-bottom:2rem;
        max-width:100% !important;
        padding-left:2rem;
        padding-right:2rem;
    }

    h1,h2,h3,h4,h5,h6,p,label,span,div {
        font-family:-apple-system,BlinkMacSystemFont,"Inter","Segoe UI",sans-serif;
    }

    h1,h2,h3,h4,h5,h6,p,label,span,div {
        color:var(--taxi-text);
    }

    .hero-card {
        background:linear-gradient(135deg,#111827 0%,#1F2937 100%);
        border:1px solid var(--taxi-border);
        border-radius:32px;
        padding:34px 36px;
        margin-bottom:24px;
        box-shadow:0 24px 70px rgba(0,0,0,.50);
        position:relative;
        overflow:hidden;
        color:var(--taxi-text);
    }

    .hero-card:after {
        content:"";
        position:absolute;
        height:76px;
        width:64%;
        right:-60px;
        top:-10px;
        background:repeating-linear-gradient(
            45deg,
            #FACC15 0,
            #FACC15 14px,
            #111827 14px,
            #111827 28px
        );
        border-radius:0 0 0 42px;
        transform:skewX(-28deg);
        opacity:.55;
        box-shadow:0 18px 55px rgba(250,204,21,.18);
    }

    .hero-eyebrow {
        color:var(--taxi-yellow);
        font-size:.78rem;
        font-weight:950;
        letter-spacing:.16em;
        text-transform:uppercase;
        margin-bottom:10px;
    }

    .hero-title {
        color:#FFFFFF;
        font-size:2.55rem;
        line-height:1.02;
        font-weight:950;
        margin:0;
        letter-spacing:-.045em;
    }

    .hero-subtitle {
        color:rgba(255,255,255,.76);
        font-size:1.04rem;
        max-width:760px;
        margin-top:14px;
        margin-bottom:0;
        line-height:1.55;
        font-weight:600;
    }

    .pill-row { margin-top:21px; }

    .pill {
        display:inline-block;
        background:rgba(250,204,21,.13);
        color:#FACC15;
        border:1px solid rgba(250,204,21,.32);
        padding:8px 13px;
        border-radius:999px;
        font-size:.84rem;
        font-weight:850;
        margin-right:8px;
        margin-bottom:8px;
    }

    .section-card {
        background:rgba(21,24,33,.92);
        color:var(--taxi-text);
        border:1px solid var(--taxi-border);
        border-radius:26px;
        padding:23px;
        box-shadow:0 18px 48px rgba(0,0,0,.32);
        margin-top:12px;
        margin-bottom:17px;
        backdrop-filter:blur(12px);
    }

    .metric-card {
        background:#20232D;
        border:1px solid var(--taxi-border);
        border-radius:24px;
        padding:19px;
        box-shadow:0 18px 48px rgba(0,0,0,.32);
        min-height:120px;
        color:var(--taxi-text);
    }

    .metric-label {
        font-size:.74rem;
        color:var(--taxi-muted);
        font-weight:900;
        text-transform:uppercase;
        letter-spacing:.08em;
        margin-bottom:9px;
    }

    .metric-value {
        font-size:1.95rem;
        font-weight:950;
        color:#FFFFFF;
        line-height:1.05;
        letter-spacing:-.03em;
        overflow-wrap:anywhere;
    }

    .metric-help {
        color:var(--taxi-muted);
        font-size:.84rem;
        margin-top:8px;
        font-weight:650;
    }

    .final-fare-card {
        background:linear-gradient(135deg,#FACC15 0%,#FFE580 100%);
        border:1px solid rgba(250,204,21,.68);
        box-shadow:0 18px 45px rgba(250,204,21,.16);
    }

    .final-fare-card .metric-label,
    .final-fare-card .metric-value,
    .final-fare-card .metric-help {
        color:#111827;
    }

    .soft-note {
        background:rgba(250,204,21,.12);
        border:1px solid rgba(250,204,21,.32);
        color:rgba(255,255,255,.90);
        border-radius:18px;
        padding:14px 18px;
        font-size:.94rem;
        margin-top:10px;
        line-height:1.45;
        box-shadow:0 8px 20px rgba(0,0,0,.20);
    }

    .status-chip {
        display:inline-block;
        padding:8px 12px;
        border-radius:999px;
        background:rgba(255,255,255,.07);
        border:1px solid rgba(255,255,255,.12);
        color:rgba(255,255,255,.88);
        font-size:.84rem;
        font-weight:850;
        margin-right:6px;
        margin-bottom:7px;
    }

    .stButton > button {
        background:#FACC15;
        color:#111827;
        border:2px solid #FACC15;
        border-radius:999px;
        padding:.68rem 1.25rem;
        font-weight:950;
        box-shadow:0 10px 24px rgba(0,0,0,.24);
        transition:all .18s ease;
    }

    .stButton > button:hover {
        background:#111827;
        color:#FACC15;
        border:2px solid #FACC15;
        transform:translateY(-1px);
    }

    .stButton > button[kind="primary"] {
        background:#FACC15;
        color:#111827;
        border:2px solid #FACC15;
    }

    .stButton > button[kind="primary"]:hover {
        background:#111827;
        color:#FACC15;
    }

    input,textarea,[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"] {
        color:#FFFFFF !important;
    }

    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div,
    [data-baseweb="select"] > div {
        background:#111827 !important;
        border-color:rgba(250,204,21,.35) !important;
        border-radius:14px !important;
        color:#FFFFFF !important;
    }

    [data-baseweb="select"] span,
    [data-baseweb="select"] input,
    [data-baseweb="select"] svg {
        color:#FFFFFF !important;
        fill:#FFFFFF !important;
    }

    div[data-baseweb="popover"],
    div[data-baseweb="menu"] {
        background:#111827 !important;
        color:#FFFFFF !important;
        border:1px solid rgba(250,204,21,.35) !important;
    }

    div[data-baseweb="option"] {
        background:#111827 !important;
        color:#FFFFFF !important;
    }

    div[data-baseweb="option"] *,
    div[data-baseweb="option"] div,
    div[data-baseweb="option"] span {
        color:#FFFFFF !important;
    }

    div[data-baseweb="option"]:hover,
    div[data-baseweb="option"][aria-selected="true"] {
        background:#FACC15 !important;
        color:#111827 !important;
    }

    div[data-baseweb="option"]:hover *,
    div[data-baseweb="option"][aria-selected="true"] *,
    div[data-baseweb="option"]:hover div,
    div[data-baseweb="option"][aria-selected="true"] div,
    div[data-baseweb="option"]:hover span,
    div[data-baseweb="option"][aria-selected="true"] span {
        color:#111827 !important;
    }

    .stSlider [data-baseweb="slider"] div { color:#FACC15 !important; }

    div[data-testid="stTabs"] button p {
        font-weight:900;
        font-size:.98rem;
        color:rgba(255,255,255,.70);
    }

    div[data-testid="stTabs"] button[aria-selected="true"] p { color:#FACC15; }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color:#FACC15 !important; }

    [data-testid="stSidebar"] {
        background:#05070A;
        border-right:1px solid var(--taxi-border);
    }

    [data-testid="stSidebar"] * { color:rgba(255,255,255,.88) !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#FACC15 !important; }

    [data-testid="stMetric"] {
        background:rgba(255,255,255,.055);
        border:1px solid rgba(255,255,255,.10);
        border-radius:18px;
        padding:13px 14px;
    }

    [data-testid="stMetricLabel"] { color:var(--taxi-muted) !important; }
    [data-testid="stMetricValue"] { color:#FFFFFF !important; }

    [data-testid="stChatMessage"] {
        background:rgba(255,255,255,.055);
        border:1px solid rgba(255,255,255,.10);
        border-radius:22px;
        padding:12px 14px;
        box-shadow:0 12px 30px rgba(0,0,0,.22);
    }

    [data-testid="stChatInput"] textarea {
        background:#111827 !important;
        color:#FFFFFF !important;
        border-radius:22px !important;
        border:1px solid rgba(250,204,21,.35) !important;
    }

    .stDataFrame,iframe { border-radius:18px; overflow:hidden; }
    hr { border-color:rgba(255,255,255,.12); }

    div[data-testid="stElementContainer"] > div:has(canvas) {
        width: 100vw !important;
        max-width: 100vw !important;
        margin-left: calc(-50vw + 50%) !important;
        margin-right: calc(-50vw + 50%) !important;
    }
</style>
<div class="hero-card">
    <div class="hero-eyebrow">NYC TLC Fare Simulation System</div>
    <h1 class="hero-title">NYC Yellow Taxi Fare Prediction</h1>
    <p class="hero-subtitle">Estimate NYC Yellow Taxi fares using pickup/drop-off zones, understand price changes, and simulate demand-aware pricing adjustments under regulated taxi conditions.</p>
    <div class="pill-row"><span class="pill">Pricing adjustment simulation</span><span class="pill">Weather + traffic aware</span><span class="pill">Explainable output</span></div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PROJECT CONTEXT
# ============================================================
PROJECT_CONTEXT = """
Project title:
NYC Yellow Taxi Fare Prediction

Project scope:
- NYC Yellow Taxi fare prediction using machine learning
- Demand-aware pricing adjustment simulation under regulated taxi fare structure
- Weather, traffic, and airport impact analysis
- Interactive trip estimation and project Q&A

Data:
- January 2024 NYC Yellow Taxi trip data
- NYC weather data aggregated to daily level

Main findings:
- Best overall fare model: XGBoost
- Demand forecasting R2 is about 0.71
- Combined pricing strategy produced about 2.6% revenue lift
- Traffic-based adjustments contributed the most among the tested strategies
- Evening demand is higher, especially around 5 PM to 6 PM
- Congestion is worse during peak hours
- Airport trips generally have higher fare and higher revenue per hour
- Limitation: taxi data is from a single month, so long-term seasonal generalization is limited

Rules:
- Answer only using this project context and any trip output provided by the app
- Do not invent unsupported metrics
- If the project does not measure something directly, say so clearly
- Be concise, clear, and practical
- If a route_note is provided, include it naturally inside the explanation. Do not create a separate Route Estimate Note heading.
"""

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "fare_model.pkl")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Could not load fare_model.pkl. Error: {e}")
        st.stop()

@st.cache_resource
def load_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None

model = load_model()
client = load_openai_client()

# ============================================================
# HELPERS
# ============================================================
def get_expected_columns(model_obj):
    if hasattr(model_obj, "feature_names_in_"):
        cols = list(model_obj.feature_names_in_)
        if cols:
            return cols

    if hasattr(model_obj, "named_steps"):
        for _, step in model_obj.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                cols = list(step.feature_names_in_)
                if cols:
                    return cols
    return None


EXPECTED_COLUMNS = get_expected_columns(model)


# ============================================================
# LOCATION PARSING + DISTANCE ESTIMATION
# ============================================================
BOROUGH_DEFAULTS = {
    "manhattan": {"id": 230, "label": "Times Sq/Theatre District", "lat": 40.7580, "lon": -73.9855},
    "brooklyn": {"id": 65, "label": "Downtown Brooklyn/MetroTech", "lat": 40.6930, "lon": -73.9857},
    "queens": {"id": 145, "label": "Long Island City/Hunters Point", "lat": 40.7440, "lon": -73.9488},
    "bronx": {"id": 42, "label": "Central Harlem North", "lat": 40.8143, "lon": -73.9400},
    "staten island": {"id": 5, "label": "Arden Heights", "lat": 40.5520, "lon": -74.1730},
    "newark": {"id": 1, "label": "Newark Airport", "lat": 40.6895, "lon": -74.1745},
    "ewr": {"id": 1, "label": "Newark Airport", "lat": 40.6895, "lon": -74.1745},
}

CURATED_ZONE_CENTROIDS = {
    1: {"lat": 40.6895, "lon": -74.1745},
    33: {"lat": 40.6958, "lon": -73.9956},
    65: {"lat": 40.6930, "lon": -73.9857},
    66: {"lat": 40.7033, "lon": -73.9881},
    87: {"lat": 40.7075, "lon": -74.0113},
    138: {"lat": 40.7769, "lon": -73.8740},
    132: {"lat": 40.6413, "lon": -73.7781},
    145: {"lat": 40.7440, "lon": -73.9488},
    161: {"lat": 40.7549, "lon": -73.9840},
    162: {"lat": 40.7540, "lon": -73.9708},
    181: {"lat": 40.6720, "lon": -73.9770},
    211: {"lat": 40.7233, "lon": -74.0030},
    230: {"lat": 40.7580, "lon": -73.9855},
    236: {"lat": 40.7736, "lon": -73.9566},
    239: {"lat": 40.7865, "lon": -73.9754},
    255: {"lat": 40.7181, "lon": -73.9571},
}


TLC_TAXI_ZONES_ZIP_URL = "https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip"

@st.cache_data(show_spinner=False)
def load_geometry_centroids():
    """Return {LocationID: {lat, lon}} from official TLC taxi-zone geometry.

    This does NOT require the user to manually download a file. The app first checks
    for local geometry files, then tries to download the official TLC taxi_zones.zip.
    If geopandas/requests/internet are unavailable, it safely falls back to {}.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "taxi_zones.zip"),
        os.path.join(current_dir, "taxi_zones.geojson"),
        os.path.join(current_dir, "taxi_zones", "taxi_zones.shp"),
    ]

    if gpd is None:
        return {}, "geopandas is not installed, so official centroid distance is unavailable. Install it with: pip install geopandas pyogrio requests"

    geometry_path = next((p for p in candidates if os.path.exists(p)), None)

    if geometry_path is None:
        geometry_path = os.path.join(current_dir, "taxi_zones.zip")
        try:
            import requests
            response = requests.get(TLC_TAXI_ZONES_ZIP_URL, timeout=25)
            response.raise_for_status()
            with open(geometry_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            return {}, f"Official geometry not found locally and auto-download failed: {e}"

    try:
        gdf = gpd.read_file(geometry_path)

        lower_cols = {str(c).lower(): c for c in gdf.columns}
        id_col = None
        for candidate in ["locationid", "location_id", "objectid"]:
            if candidate in lower_cols:
                id_col = lower_cols[candidate]
                break
        if id_col is None:
            return {}, f"Geometry loaded, but no LocationID column was found. Columns: {list(gdf.columns)}"

        gdf = gdf.dropna(subset=[id_col]).copy()
        gdf[id_col] = gdf[id_col].astype(int)

        # TLC shapefile is commonly EPSG:2263. If CRS is missing, assume that.
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=2263)

        # Compute centroids in a projected CRS, then convert centroid points to lat/lon.
        projected = gdf.to_crs(epsg=2263)
        centroids = projected.geometry.centroid
        centroid_gdf = gpd.GeoDataFrame(projected[[id_col]].copy(), geometry=centroids, crs="EPSG:2263").to_crs(epsg=4326)

        result = {}
        for _, row in centroid_gdf.iterrows():
            loc_id = int(row[id_col])
            result[loc_id] = {"lat": float(row.geometry.y), "lon": float(row.geometry.x)}

        return result, f"Loaded official TLC taxi-zone geometry for {len(result)} zones from {os.path.basename(geometry_path)}."
    except Exception as e:
        return {}, f"Could not read taxi-zone geometry: {e}"

GEOMETRY_ZONE_CENTROIDS, GEOMETRY_STATUS = load_geometry_centroids()

def normalize_location_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("&", " and ")

    # Common user spelling variations / shortcuts
    text = re.sub(r"\bla\s+guardia\b", "laguardia", text)
    text = re.sub(r"\bl\s*g\s*a\b", "lga", text)
    text = re.sub(r"\bj\s*f\s*k\b", "jfk", text)
    text = re.sub(r"\bnyc\b", "new york city", text)

    text = re.sub(r"[()]", " ", text)
    text = re.sub(r"[^a-z0-9\s/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def simplify_zone_name(name: str) -> str:
    name = normalize_location_text(name)
    name = name.replace(" / ", "/").replace("/ ", "/").replace(" /", "/")
    return name

@st.cache_data
def load_zone_lookup():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "Pasted text(12).txt"),
        os.path.join(current_dir, "taxi_zone_lookup.csv"),
    ]
    zone_path = next((p for p in candidates if os.path.exists(p)), None)
    if zone_path is None:
        return pd.DataFrame(columns=["LocationID", "Borough", "Zone", "service_zone"])

    try:
        zones = pd.read_csv(zone_path)
    except Exception:
        return pd.DataFrame(columns=["LocationID", "Borough", "Zone", "service_zone"])

    zones.columns = [str(c).strip() for c in zones.columns]
    needed = {"LocationID", "Borough", "Zone"}
    if not needed.issubset(set(zones.columns)):
        return pd.DataFrame(columns=["LocationID", "Borough", "Zone", "service_zone"])

    zones = zones.copy()
    zones["LocationID"] = pd.to_numeric(zones["LocationID"], errors="coerce").astype("Int64")
    zones = zones.dropna(subset=["LocationID", "Borough", "Zone"]).copy()
    zones["LocationID"] = zones["LocationID"].astype(int)
    zones["Borough"] = zones["Borough"].astype(str)
    zones["Zone"] = zones["Zone"].astype(str)
    if "service_zone" not in zones.columns:
        zones["service_zone"] = ""
    return zones

def build_zone_aliases():
    zones = load_zone_lookup()
    alias_map = {}

    def add_alias(alias: str, payload: dict):
        alias = normalize_location_text(alias)
        if not alias or alias in {"n a", "unknown", "outside of nyc"}:
            return
        alias_map.setdefault(alias, payload)

    for _, row in zones.iterrows():
        zone_id = int(row["LocationID"])
        borough = normalize_location_text(str(row["Borough"]))
        zone = str(row["Zone"]).strip()
        zone_norm = simplify_zone_name(zone)

        borough_default = BOROUGH_DEFAULTS.get(borough, BOROUGH_DEFAULTS.get("manhattan"))
        centroid = GEOMETRY_ZONE_CENTROIDS.get(zone_id) or CURATED_ZONE_CENTROIDS.get(zone_id, {"lat": borough_default["lat"], "lon": borough_default["lon"]})

        payload = {
            "id": zone_id,
            "label": zone,
            "borough": borough,
            "lat": centroid["lat"],
            "lon": centroid["lon"],
            "granularity": "place",
        }

        add_alias(zone, payload)
        add_alias(zone_norm, payload)
        add_alias(zone_norm.replace("/", " "), payload)
        add_alias(zone_norm.replace("/", " / "), payload)

        no_paren = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", " ", zone_norm)).strip()
        if no_paren and no_paren != zone_norm:
            add_alias(no_paren, payload)

        for part in [p.strip() for p in zone_norm.split("/") if p.strip()]:
            if len(part) >= 3:
                add_alias(part, payload)
                add_alias(f"{borough} {part}", payload)

        add_alias(f"{borough} {zone_norm}", payload)

    # Friendly manual aliases
    manual_aliases = {
        "times square": 230,
        "theatre district": 230,
        "midtown": 161,
        "midtown center": 161,
        "midtown east": 162,
        "midtown west": 230,
        "downtown brooklyn": 65,
        "metrotech": 65,
        "financial district": 87,
        "wall street": 87,
        "upper east side": 236,
        "upper west side": 239,
        "long island city": 145,
        "lic": 145,
        "williamsburg": 255,
        "dumbo": 66,
        "park slope": 181,
        "soho": 211,
        "jfk": 132,
        "jfk airport": 132,
        "laguardia": 138,
        "laguardia airport": 138,
        "la guardia": 138,
        "la guardia airport": 138,
        "lga": 138,
        "newark airport": 1,
    }

    for alias, zone_id in manual_aliases.items():
        match = next((v for v in alias_map.values() if v["id"] == zone_id), None)
        if match:
            add_alias(alias, match)

    return alias_map

ZONE_ALIASES = build_zone_aliases()
AREA_KEYWORDS = sorted(set(list(ZONE_ALIASES.keys()) + list(BOROUGH_DEFAULTS.keys())), key=len, reverse=True)

def lookup_area(area_text: str):
    if not area_text:
        return None

    area_text = normalize_location_text(area_text)

    if area_text in ZONE_ALIASES:
        return dict(ZONE_ALIASES[area_text])

    if area_text in BOROUGH_DEFAULTS:
        return {**BOROUGH_DEFAULTS[area_text], "borough": area_text, "granularity": "borough"}

    for key in AREA_KEYWORDS:
        if re.search(rf"\b{re.escape(key)}\b", area_text):
            if key in ZONE_ALIASES:
                return dict(ZONE_ALIASES[key])
            if key in BOROUGH_DEFAULTS:
                return {**BOROUGH_DEFAULTS[key], "borough": key, "granularity": "borough"}

    return None

def extract_route_locations(text: str):
    normalized = normalize_location_text(text)

    route_match = re.search(
        r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+at\b|\s+around\b|\s+with\b|\s+in\b|\s+on\b|\s+for\b|$)",
        normalized
    )
    if route_match:
        origin = lookup_area(route_match.group(1))
        destination = lookup_area(route_match.group(2))
        return origin, destination

    simple_match = re.search(
        r"\b(.+?)\s+to\s+(.+?)(?:\s+at\b|\s+around\b|\s+with\b|\s+in\b|\s+on\b|\s+for\b|$)",
        normalized
    )
    if simple_match:
        origin = lookup_area(simple_match.group(1))
        destination = lookup_area(simple_match.group(2))
        if origin and destination:
            return origin, destination

    found = []
    seen_ids = set()
    for key in AREA_KEYWORDS:
        if re.search(rf"\b{re.escape(key)}\b", normalized):
            match = lookup_area(key)
            if match and match.get("id") not in seen_ids:
                found.append(match)
                seen_ids.add(match.get("id"))

    if len(found) >= 2:
        return found[0], found[1]

    return None, None

def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.7613
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

def estimate_trip_distance_from_locations(origin: dict, destination: dict):
    straight_line = haversine_miles(origin["lat"], origin["lon"], destination["lat"], destination["lon"])

    has_geometry_for_both = (
        origin.get("id") in GEOMETRY_ZONE_CENTROIDS
        and destination.get("id") in GEOMETRY_ZONE_CENTROIDS
    )
    uses_borough_fallback = (
        origin.get("granularity") == "borough"
        or destination.get("granularity") == "borough"
        or (origin.get("id") not in GEOMETRY_ZONE_CENTROIDS and origin.get("id") not in CURATED_ZONE_CENTROIDS)
        or (destination.get("id") not in GEOMETRY_ZONE_CENTROIDS and destination.get("id") not in CURATED_ZONE_CENTROIDS)
    )

    road_factor = 1.28 if has_geometry_for_both else (1.35 if uses_borough_fallback else 1.22)
    estimated_road_miles = max(round(straight_line * road_factor, 2), 1.0)

    if has_geometry_for_both:
        note = f"Estimated distance from official TLC taxi-zone centroids for {origin['label']} to {destination['label']}."
    elif origin.get("granularity") == "borough" or destination.get("granularity") == "borough":
        note = (
            f"That route can vary depending on the exact neighborhoods. "
            f"I used a central estimate from {origin['label']} to {destination['label']}."
        )
    elif (origin.get("id") not in GEOMETRY_ZONE_CENTROIDS and origin.get("id") not in CURATED_ZONE_CENTROIDS) or (destination.get("id") not in GEOMETRY_ZONE_CENTROIDS and destination.get("id") not in CURATED_ZONE_CENTROIDS):
        note = None
    else:
        note = None 
    return estimated_road_miles, note
def safe_float(val, default):
    try:
        return float(val)
    except Exception:
        return default

def parse_hour_from_text(text: str):
    text = text.lower().strip()

    m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", text)
    if m:
        h = int(m.group(1))
        ap = m.group(2)
        if ap == "pm" and h != 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        return h

    m = re.search(r"\bhour\s*(=|:)?\s*(\d{1,2})\b", text)
    if m:
        h = int(m.group(2))
        if 0 <= h <= 23:
            return h

    return None



def parse_fuzzy_time(text: str):
    """Map natural-language time phrases to a representative pickup hour."""
    text = normalize_location_text(text)

    # More specific phrases first
    fuzzy_map = [
        (["early morning"], 7),
        (["morning"], 9),
        (["noon", "midday", "mid day"], 12),
        (["early afternoon"], 13),
        (["afternoon"], 14),
        (["evening", "around evening"], 17),
        (["rush hour", "peak hour", "peak time"], 17),
        (["late evening"], 20),
        (["late night"], 23),
        (["night"], 21),
        (["midnight"], 0),
    ]

    for phrases, hour in fuzzy_map:
        if any(phrase in text for phrase in phrases):
            return hour
    return None

def detect_trip_query(user_text: str) -> bool:
    text = user_text.lower()

    travel_phrases = [
        "travel from", "going from", "go from", "want to go from", "wanna go from",
        "ride from", "trip from", "from ", " to ",
        "taxi", "cab", "ride", "fare", "price", "cost", "how much",
        "airport", "mile", "miles", "mi"
    ]

    return any(phrase in text for phrase in travel_phrases)

def detect_time_optimization_query(user_text: str) -> bool:
    """Detect questions where the user wants the cheapest/best hour, not one fixed-hour fare."""
    text = user_text.lower()
    keywords = [
        "best time", "cheapest time", "lowest fare", "low fare",
        "least expensive", "save money", "when should i travel",
        "when to travel", "what time should i go", "best hour",
        "cheapest hour", "avoid peak", "avoid rush", "time to travel"
    ]
    return any(k in text for k in keywords)

def bucket_distance(miles: float) -> str:
    if miles < 2:
        return "short"
    elif miles < 5:
        return "medium"
    elif miles < 10:
        return "long"
    return "very_long"

def estimate_duration(trip_distance, pickup_hour, precip=0.0, windgust=20.0, is_airport=0):
    base_speed = 18.0

    is_rush_hour = int(pickup_hour in [7, 8, 9, 16, 17, 18])
    is_night = int(pickup_hour >= 22 or pickup_hour <= 5)

    if is_rush_hour:
        base_speed *= 0.70

    if is_night:
        base_speed *= 1.15

    if precip > 5:
        base_speed *= 0.85

    if windgust > 35:
        base_speed *= 0.92

    if is_airport:
        base_speed *= 0.95

    base_speed = max(base_speed, 5.0)
    duration_min = (trip_distance / base_speed) * 60.0
    return max(round(duration_min, 1), 5.0)

def extract_trip_info(user_text: str):
    text = normalize_location_text(user_text)

    info = {
        "pickup_hour": None,
        "pickup_weekday": 2,
        "trip_distance": None,
        "trip_duration_min": None,   # estimated later
        "passenger_count": 1,
        "RatecodeID": 1,
        "payment_type": 1,
        # Safe non-airport defaults. These are overwritten when a route is detected.
        # Important: do NOT default pickup to 138 (LaGuardia), because follow-up
        # messages like "2pm" can accidentally turn normal trips into airport trips.
        "PULocationID": 230,
        "DOLocationID": 65,
        "tip_amount": 2.0,
        "tolls_amount": 0.0,
        "congestion_surcharge": 2.5,
        "Airport_fee": 0.0,
        "precip": 0.0,
        "windgust": 20.0,
        "snowdepth": 0.0,
        "snow": 0.0,
        "temp": 15.0,
        "humidity": 50.0,
        "pickup_date": "2024-01-15",
        "route_note": None,
        "route_origin_label": None,
        "route_destination_label": None,
        "origin_lat": None,
        "origin_lon": None,
        "destination_lat": None,
        "destination_lon": None,
    }

    m = re.search(r"(\d+(\.\d+)?)\s*(mile|miles|mi)\b", text)
    if m:
        info["trip_distance"] = float(m.group(1))

    hour = parse_hour_from_text(text)
    if hour is None:
        hour = parse_fuzzy_time(text)
    if hour is not None:
        info["pickup_hour"] = hour

    passenger_match = re.search(r"(\d+)\s*(passenger|passengers|people|person)\b", text)
    if passenger_match:
        info["passenger_count"] = int(passenger_match.group(1))

    origin, destination = extract_route_locations(text)
    if origin:
        info["PULocationID"] = int(origin["id"])
        info["route_origin_label"] = origin["label"]
        info["origin_lat"] = origin.get("lat")
        info["origin_lon"] = origin.get("lon")
    if destination:
        info["DOLocationID"] = int(destination["id"])
        info["route_destination_label"] = destination["label"]
        info["destination_lat"] = destination.get("lat")
        info["destination_lon"] = destination.get("lon")

    if info["trip_distance"] is None and origin and destination:
        estimated_miles, route_note = estimate_trip_distance_from_locations(origin, destination)
        info["trip_distance"] = estimated_miles
        info["route_note"] = route_note

    airport_zone_ids = {1, 132, 138}
    if (
        any(word in text for word in ["airport", "jfk", "lga", "laguardia"])
        or int(info.get("PULocationID", 0)) in airport_zone_ids
        or int(info.get("DOLocationID", 0)) in airport_zone_ids
    ):
        info["Airport_fee"] = 1.25
        info["RatecodeID"] = 2

    if any(word in text for word in ["bad weather", "rain", "raining", "storm", "heavy rain"]):
        info["precip"] = 12.0
        info["windgust"] = 45.0

    if any(word in text for word in ["snow", "snowing"]):
        info["snow"] = 2.0
        info["snowdepth"] = 1.0
        info["temp"] = 1.0

    missing = []
    if info["trip_distance"] is None:
        missing.append("trip distance in miles or a recognizable route like 'Times Square to Downtown Brooklyn'")
    if info["pickup_hour"] is None:
        missing.append("pickup time, like 5 pm")

    return info, missing

def merge_partial_info(base_info: dict, user_text: str):
    new_info, _ = extract_trip_info(user_text)

    # When the user sends a follow-up like "2pm", extract_trip_info() creates
    # safe default route IDs. We should NOT let those defaults overwrite the
    # real route that was captured in the previous message. Only overwrite
    # route fields when the new message actually contains a recognizable route.
    new_has_route = (
        new_info.get("route_origin_label") is not None
        or new_info.get("route_destination_label") is not None
        or new_info.get("route_note") is not None
    )

    new_text = normalize_location_text(user_text)
    new_has_explicit_distance = re.search(r"(\d+(\.\d+)?)\s*(mile|miles|mi)\b", new_text) is not None

    for k, v in new_info.items():
        if v is None:
            continue

        if k in ["PULocationID", "DOLocationID", "route_origin_label", "route_destination_label", "route_note"] and not new_has_route:
            continue

        if k == "trip_distance" and base_info.get("trip_distance") is not None and not (new_has_route or new_has_explicit_distance):
            continue

        # Do not reset airport settings on time-only follow-ups.
        if k in ["Airport_fee", "RatecodeID"] and not new_has_route and not any(word in new_text for word in ["airport", "jfk", "lga", "laguardia"]):
            continue

        base_info[k] = v

    missing = []
    if base_info["trip_distance"] is None:
        missing.append("trip distance in miles")
    if base_info["pickup_hour"] is None:
        missing.append("pickup time, like 5 pm")

    return base_info, missing

def build_feature_row(info: dict):
    trip_distance = safe_float(info.get("trip_distance"), 3.0)
    pickup_hour = int(info.get("pickup_hour", 12))
    pickup_weekday = int(info.get("pickup_weekday", 2))

    is_airport = int(float(info.get("Airport_fee", 0.0)) > 0)

    trip_duration_min = estimate_duration(
        trip_distance=trip_distance,
        pickup_hour=pickup_hour,
        precip=safe_float(info.get("precip"), 0.0),
        windgust=safe_float(info.get("windgust"), 20.0),
        is_airport=is_airport
    )

    speed_mph = float(np.clip(trip_distance / (trip_duration_min / 60.0), 0.5, 80.0))
    is_rush_hour = int(pickup_hour in [7, 8, 9, 16, 17, 18])
    is_night = int(pickup_hour >= 22 or pickup_hour <= 5)
    is_weekend = int(pickup_weekday in [5, 6])

    bad_weather_day = int(
        safe_float(info.get("precip"), 0.0) >= 10.0
        or safe_float(info.get("windgust"), 0.0) >= 40.0
        or safe_float(info.get("snow"), 0.0) > 0
        or safe_float(info.get("snowdepth"), 0.0) > 0
    )

    demand_index = 1.0
    if is_rush_hour:
        demand_index += 0.4
    if is_weekend:
        demand_index += 0.1
    if is_airport:
        demand_index += 0.1
    demand_index = round(demand_index, 3)

    traffic_intensity = 0.6 if is_rush_hour else 0.3
    demand_strength = demand_index * 10
    demand_x_traffic = demand_index * traffic_intensity

    fare_per_mile_proxy = 3.25
    fare_per_min_proxy = 0.55

    hour_sin = math.sin(2 * math.pi * pickup_hour / 24.0)
    hour_cos = math.cos(2 * math.pi * pickup_hour / 24.0)

    row = {
        "Airport_fee": safe_float(info.get("Airport_fee"), 0.0),
        "BadWeatherDay": bad_weather_day,
        "DOLocationID": int(info.get("DOLocationID", 239)),
        "DemandIndex": demand_index,
        "PULocationID": int(info.get("PULocationID", 138)),
        "RatecodeID": int(info.get("RatecodeID", 1)),
        "congestion_surcharge": safe_float(info.get("congestion_surcharge"), 2.5),
        "demand_strength": demand_strength,
        "demand_x_traffic": demand_x_traffic,
        "distance_bucket": bucket_distance(trip_distance),
        "fare_per_mile_proxy": fare_per_mile_proxy,
        "fare_per_min_proxy": fare_per_min_proxy,
        "hour_cos": hour_cos,
        "hour_sin": hour_sin,
        "humidity": safe_float(info.get("humidity"), 50.0),
        "is_airport": is_airport,
        "is_night": is_night,
        "is_rush_hour": is_rush_hour,
        "is_weekend": is_weekend,
        "passenger_count": int(info.get("passenger_count", 1)),
        "payment_type": int(info.get("payment_type", 1)),
        "pickup_date": str(info.get("pickup_date", "2024-01-15")),
        "pickup_hour": pickup_hour,
        "pickup_weekday": pickup_weekday,
        "precip": safe_float(info.get("precip"), 0.0),
        "snow": safe_float(info.get("snow"), 0.0),
        "snowdepth": safe_float(info.get("snowdepth"), 0.0),
        "speed_mph": speed_mph,
        "temp": safe_float(info.get("temp"), 15.0),
        "tip_amount": safe_float(info.get("tip_amount"), 2.0),
        "tolls_amount": safe_float(info.get("tolls_amount"), 0.0),
        "trip_distance": trip_distance,
        "trip_duration_min": trip_duration_min,
        "traffic_intensity": traffic_intensity,
        "windgust": safe_float(info.get("windgust"), 20.0),
    }

    meta = {
        "estimated_duration_min": round(trip_duration_min, 1),
        "speed_mph": round(speed_mph, 2),
        "is_rush_hour": is_rush_hour,
        "bad_weather_day": bad_weather_day,
        "is_airport": is_airport,
        "demand_index": demand_index,
        "traffic_intensity": traffic_intensity,
        "route_note": info.get("route_note"),
        "route_origin_label": info.get("route_origin_label"),
        "route_destination_label": info.get("route_destination_label"),
        "origin_lat": info.get("origin_lat"),
        "origin_lon": info.get("origin_lon"),
        "destination_lat": info.get("destination_lat"),
        "destination_lon": info.get("destination_lon"),
    }

    return row, meta

def prepare_model_input(row: dict, model_obj):
    df = pd.DataFrame([row])

    expected_cols = get_expected_columns(model_obj)
    if expected_cols:
        for col in expected_cols:
            if col not in df.columns:
                if col == "distance_bucket":
                    df[col] = "unknown"
                elif col == "pickup_date":
                    df[col] = "2024-01-15"
                else:
                    df[col] = 0
        df = df[expected_cols]

    return df


def pricing_adjustment(base_fare: float, meta: dict):
    """Small regulated-policy simulation layer for NYC Yellow Taxi analysis.

    This is not an official TLC fare rule; it is a project simulation layer.
    It is a capstone simulation used to test how demand, traffic, weather,
    and airport conditions could affect the estimated fare/revenue.
    """
    demand_add = 0.0
    traffic_add = 0.0
    weather_add = 0.0
    airport_add = 0.0

    if meta["demand_index"] >= 1.4:
        demand_add = 1.00

    if meta["traffic_intensity"] >= 0.5:
        traffic_add = 1.25

    if meta["bad_weather_day"]:
        weather_add = 0.50

    if meta["is_airport"]:
        airport_add = 0.75

    adjusted = (base_fare + demand_add + traffic_add + weather_add + airport_add) * 1.03

    breakdown = {
        "base_fare": round(base_fare, 2),
        "demand_add": round(demand_add, 2),
        "traffic_add": round(traffic_add, 2),
        "weather_add": round(weather_add, 2),
        "airport_add": round(airport_add, 2),
        "final_fare": round(adjusted, 2),
    }
    return adjusted, breakdown

       
def fallback_trip_response(base_fare, adjusted_fare, meta, breakdown, route_note=None):
    reasons = []
    if meta["is_rush_hour"]:
        reasons.append("rush-hour demand and congestion are increasing the estimate")
    if meta["traffic_intensity"] >= 0.5:
        reasons.append("traffic conditions add a small adjustment")
    if meta["bad_weather_day"]:
        reasons.append("bad weather increases pricing pressure")
    if meta["is_airport"]:
        reasons.append("an airport premium is included")

    if reasons:
        reason_sentence = "The main reason is that " + ", and ".join(reasons) + "."
    else:
        reason_sentence = "This looks like a standard trip, so no major demand, weather, traffic, or airport surcharge was added."

    route_sentence = f"\n\n{route_note}" if route_note else ""

    return f"""
Estimated final fare: \\${adjusted_fare:.2f}  
Base model fare: \\${base_fare:.2f} · Estimated duration: {meta['estimated_duration_min']} min

Pricing adjustments: demand \\${breakdown['demand_add']:.2f}, traffic \\${breakdown['traffic_add']:.2f}, weather \\${breakdown['weather_add']:.2f}, airport \\${breakdown['airport_add']:.2f}.

{reason_sentence}{route_sentence}

Trip details: estimated speed {meta['speed_mph']} mph, rush hour {'yes' if meta['is_rush_hour'] else 'no'}, bad weather {'yes' if meta['bad_weather_day'] else 'no'}, airport trip {'yes' if meta['is_airport'] else 'no'}.
"""

def llm_answer(messages, system_prompt):
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                *messages
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        st.sidebar.warning(f"LLM unavailable: {e}")
        return None

def run_prediction_from_info(info: dict):
    row, meta = build_feature_row(info)
    input_df = prepare_model_input(row, model)
    base_pred = float(model.predict(input_df)[0])
    adjusted_pred, breakdown = pricing_adjustment(base_pred, meta)
    return input_df, meta, base_pred, adjusted_pred, breakdown


def optimize_best_travel_time(info: dict):
    """Run the same route across all 24 pickup hours and identify the cheapest option."""
    results = []
    for hour in range(24):
        test_info = info.copy()
        test_info["pickup_hour"] = hour
        _, meta, base_pred, adjusted_pred, breakdown = run_prediction_from_info(test_info)
        results.append({
            "hour": hour,
            "time_label": f"{hour % 12 or 12} {'AM' if hour < 12 else 'PM'}",
            "base_fare": round(base_pred, 2),
            "final_fare": round(adjusted_pred, 2),
            "duration_min": meta["estimated_duration_min"],
            "rush_hour": "Yes" if meta["is_rush_hour"] else "No",
            "traffic_intensity": meta["traffic_intensity"],
            "demand_index": meta["demand_index"],
            "demand_add": breakdown["demand_add"],
            "traffic_add": breakdown["traffic_add"],
            "weather_add": breakdown["weather_add"],
            "airport_add": breakdown["airport_add"],
        })
    result_df = pd.DataFrame(results)
    best_row = result_df.loc[result_df["final_fare"].idxmin()]
    worst_row = result_df.loc[result_df["final_fare"].idxmax()]
    savings = float(worst_row["final_fare"] - best_row["final_fare"])
    return result_df, best_row, worst_row, savings

def build_time_optimization_reply(info: dict, best_row, worst_row, savings):
    route_text = ""
    if info.get("route_origin_label") and info.get("route_destination_label"):
        route_text = f" from {info['route_origin_label']} to {info['route_destination_label']}"
    return f"""
Best time to travel{route_text}: {best_row['time_label']}

Estimated lowest fare: ${best_row['final_fare']:.2f}  
Most expensive time: {worst_row['time_label']} at ${worst_row['final_fare']:.2f}  
Potential saving by avoiding the peak time: ${savings:.2f}

Why this time is cheaper:
- Lower rush-hour pressure
- Lower traffic adjustment
- Lower demand index compared with peak hours

Estimate based on typical NYC taxi conditions including traffic, demand, and weather. This is not an official TLC fare quote.
"""



def detect_fare_estimate_query(user_text: str) -> bool:
    """Detect questions where the user is asking for a price estimate, not just service availability."""
    text = user_text.lower()
    keywords = [
        "fare", "cost", "price", "how much", "estimate", "estimated",
        "what would be", "what will be", "how expensive", "charge", "pay"
    ]
    return any(k in text for k in keywords)


def detect_service_query(user_text: str) -> bool:
    """Detect service-availability questions without hijacking fare-estimate questions."""
    text = user_text.lower()

    if detect_fare_estimate_query(user_text):
        return False

    # Only explicit service/availability questions should enter this branch.
    # Do NOT treat phrases like "to JFK" or "from JFK" as service questions,
    # because normal fare requests such as "I want to go from Manhattan to JFK at 5 PM"
    # must go to the fare prediction branch.
    keywords = [
        "service available", "is service available", "service to", "service from",
        "is there service", "is there a taxi", "is taxi available", "taxi available",
        "do taxis go", "does taxi go", "can i take a taxi", "can i get a taxi",
        "available to", "available from"
    ]
    return any(k in text for k in keywords)

def has_explicit_from_to_route(prompt: str) -> bool:
    """True when the user clearly gives both an origin and destination using 'from ... to ...'."""
    text = normalize_location_text(prompt)
    return re.search(r"\bfrom\s+.+?\s+to\s+.+", text) is not None


def unsupported_route_reply(origin, destination):
    """Return a professional message when one or both locations are outside the NYC taxi-zone lookup."""
    if origin is None and destination is None:
        detail = "Both the pickup and drop-off locations are outside the supported NYC taxi zone lookup."
    elif origin is None:
        detail = "The pickup location is outside the supported NYC taxi zone lookup."
    else:
        detail = "The drop-off location is outside the supported NYC taxi zone lookup."

    return f"""
Service availability

No — this route is outside the supported NYC Yellow Taxi service area.

{detail}

This estimator only supports routes that can be mapped to recognized NYC TLC taxi zones or supported airport zones.

Try a route such as Manhattan to JFK at 5 PM or Times Square to Brooklyn in evening.
""", {}


def airport_service_reply(airport_name: str, example: str):
    return f"""
Service availability: {airport_name}

Yes — NYC Yellow Taxi service is available to and from {airport_name}.

Airport-related fees and surcharges may apply. Fare can vary based on distance, traffic, time of day, and weather conditions.

For an estimate, try: {example}.
""", {}


def handle_service_availability(prompt: str):
    """Answer NYC taxi service-availability questions using strict NYC zone validation."""
    text = prompt.lower()

    if has_explicit_from_to_route(prompt):
        origin, destination = extract_route_locations(prompt)
        if origin is None or destination is None:
            return unsupported_route_reply(origin, destination)

        origin_label = origin.get("label", "pickup")
        dest_label = destination.get("label", "drop-off")
        return f"""
Service availability: {origin_label} to {dest_label}

Yes — NYC Yellow Taxi service is available for this route.

Fare will depend on exact pickup and drop-off zones, time of day, traffic, weather, and airport-related fees if applicable.

For an estimate, try: {origin_label} to {dest_label} at 5 PM.
""", {}

    if "newark" in text or "ewr" in text:
        return airport_service_reply("Newark Airport (EWR)", "Manhattan to Newark at 5 PM")

    if "jfk" in text:
        return airport_service_reply("JFK Airport", "Manhattan to JFK at 5 PM")

    if "laguardia" in text or "lga" in text:
        return airport_service_reply("LaGuardia Airport (LGA)", "Brooklyn to LGA at 6 PM")

    return """
Service availability

NYC Yellow Taxi service is available across recognized NYC TLC taxi zones and supported airport routes.

For best results, include a pickup area, drop-off area, and time. Example: Manhattan to Brooklyn in evening.
""", {}


def detect_route_query(user_text: str) -> bool:
    """Detect route/service existence questions that should not require pickup time."""
    text = user_text.lower()
    keywords = [
        "is there a route", "route from", "routes from",
        "can i go from", "can i travel from", "is it possible to go",
        "do taxis go from", "does taxi go from", "is there taxi from",
        "available between", "taxi between"
    ]
    return any(k in text for k in keywords)


def handle_route_query(prompt: str):
    """Answer route-availability questions without forcing a fare estimate."""
    origin, destination = extract_route_locations(prompt)

    if origin is None or destination is None:
        return unsupported_route_reply(origin, destination)

    origin_label = origin.get("label", "pickup")
    dest_label = destination.get("label", "drop-off")

    return f"""
Route availability: {origin_label} to {dest_label}

Yes — NYC Yellow Taxi service is available for this route.

Fare will depend on:
- Exact pickup and drop-off neighborhoods
- Travel time
- Traffic conditions
- Weather and airport-related factors, if applicable

For a fare estimate, include a time phrase such as: {origin_label} to {dest_label} at 5 PM.
""", {}


def detect_advanced_intent(user_text: str):
    """Route advanced questions by priority. Cost questions should not be swallowed by service availability."""
    text = user_text.lower()

    if any(k in text for k in ["compare", " vs ", " versus ", "difference between"]):
        return "compare_trips"
    if any(k in text for k in ["if i travel", "instead of", "shift", "change time"]):
        return "time_savings"
    if any(k in text for k in ["best time", "cheapest time", "lowest fare", "least expensive", "when should i travel", "when to travel", "what time should i go", "best hour", "cheapest hour"]):
        return "best_time"
    if any(k in text for k in ["worst time", "most expensive time", "highest fare", "peak time"]):
        return "worst_time"
    if any(k in text for k in ["how can i make", "make this trip cheaper", "reduce fare", "save money", "how to save", "cheaper"]):
        return "save_money"

    # Direct fare estimates should fall through to the normal trip-prediction branch.
    if detect_fare_estimate_query(user_text):
        return None

    if detect_service_query(user_text):
        return "service_availability"
    if detect_route_query(user_text):
        return "route_availability"

    if any(k in text for k in ["why expensive", "why high", "why is it high", "why is this expensive", "why cost", "why does it cost"]):
        return "why_expensive"
    if any(k in text for k in ["what if it rains", "rain vs", "weather impact", "with rain", "without rain", "normal vs rain"]):
        return "weather_compare"
    if any(k in text for k in ["airport trip", "airport more expensive", "is jfk more expensive", "is airport expensive"]):
        return "airport_explain"
    if any(k in text for k in ["what affects", "main factor", "impact most", "affects price", "price factors"]):
        return "feature_importance"
    if any(k in text for k in ["long route", "congested", "traffic heavy", "route analysis", "is this route"]):
        return "route_analysis"
    if any(k in text for k in ["distance increase", "longer trip", "fare change from", "distance impact"]):
        return "distance_sensitivity"
    return None


def format_route_text(info: dict) -> str:
    if info.get("route_origin_label") and info.get("route_destination_label"):
        return f" from {info['route_origin_label']} to {info['route_destination_label']}"
    return ""


def missing_for_intent(info: dict, missing: list, intent: str) -> list:
    filtered = list(missing)
    if intent in {"best_time", "worst_time", "save_money"}:
        filtered = [m for m in filtered if "pickup time" not in m]
    if intent in {"feature_importance", "service_availability"}:
        filtered = []
    return filtered


def handle_best_time(info: dict):
    df, best, worst, savings = optimize_best_travel_time(info)
    return build_time_optimization_reply(info, best, worst, savings), {"time_df": df}


def handle_worst_time(info: dict):
    df, best, worst, savings = optimize_best_travel_time(info)
    route_text = format_route_text(info)
    reply = f"""
Worst time to travel{route_text}: {worst['time_label']}

Estimated highest fare: ${worst['final_fare']:.2f}  
Cheapest time: {best['time_label']} at ${best['final_fare']:.2f}  
Extra cost compared with the cheapest hour: ${savings:.2f}

Why it is worse:
- Higher traffic intensity
- Higher demand index
- Rush-hour pressure when applicable
"""
    return reply, {"time_df": df}


def handle_save_money(info: dict):
    df, best, worst, savings = optimize_best_travel_time(info)
    route_text = format_route_text(info)
    reply = f"""
How to make this trip cheaper{route_text}

Best option: {best['time_label']} at about ${best['final_fare']:.2f}  
Avoid: {worst['time_label']} at about ${worst['final_fare']:.2f}  
Potential saving: ${savings:.2f}

Practical suggestions:
- Avoid evening rush hour, especially around 4–6 PM.
- Prefer lower-demand hours where the model predicts lower traffic and demand pressure.
- Avoid rain/bad-weather windows if possible.
"""
    return reply, {"time_df": df}


def handle_time_savings(user_text: str, info: dict):
    text = user_text.lower()
    mentioned = []
    for m in re.finditer(r"\b(\d{1,2})\s*(am|pm)\b", text):
        h = int(m.group(1))
        ap = m.group(2)
        if ap == "pm" and h != 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        if 0 <= h <= 23:
            mentioned.append(h)
    if len(mentioned) < 2:
        mentioned = [14, 17]

    rows = []
    for h in mentioned[:2]:
        test_info = info.copy()
        test_info["pickup_hour"] = h
        _, meta, base, adj, breakdown = run_prediction_from_info(test_info)
        rows.append({
            "hour": h,
            "time_label": f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}",
            "final_fare": round(adj, 2),
            "base_fare": round(base, 2),
            "duration_min": meta["estimated_duration_min"],
            "rush_hour": "Yes" if meta["is_rush_hour"] else "No",
            "traffic_intensity": meta["traffic_intensity"],
            "demand_index": meta["demand_index"],
        })

    diff = rows[1]["final_fare"] - rows[0]["final_fare"]
    cheaper = rows[0] if rows[0]["final_fare"] <= rows[1]["final_fare"] else rows[1]
    reply = f"""
Time-shift savings comparison{format_route_text(info)}

{rows[0]['time_label']}: ${rows[0]['final_fare']:.2f}  
{rows[1]['time_label']}: ${rows[1]['final_fare']:.2f}

Cheaper option: {cheaper['time_label']}  
Estimated saving: ${abs(diff):.2f}

Reason: the more expensive option has higher demand/traffic pressure in the simulation.
"""
    return reply, {"table_df": pd.DataFrame(rows)}


def handle_why_expensive(info: dict):
    _, meta, base, adjusted, breakdown = run_prediction_from_info(info)
    reasons = []
    if meta["is_rush_hour"]:
        reasons.append("rush-hour demand is active")
    if meta["traffic_intensity"] >= 0.5:
        reasons.append("traffic intensity is high")
    if meta["bad_weather_day"]:
        reasons.append("bad weather is increasing pressure")
    if meta["is_airport"]:
        reasons.append("airport-related pricing is included")
    if not reasons:
        reasons.append("there are no major extra pressure factors; distance and estimated duration are the main drivers")

    reply = f"""
Why this fare is high{format_route_text(info)}

Estimated final fare: ${adjusted:.2f}  
Base model fare: ${base:.2f}

Main reasons:
- """ + "\n- ".join(reasons) + f"""

Adjustment breakdown:
- Demand: ${breakdown['demand_add']:.2f}
- Traffic: ${breakdown['traffic_add']:.2f}
- Weather: ${breakdown['weather_add']:.2f}
- Airport: ${breakdown['airport_add']:.2f}
"""
    return reply, {"meta": meta, "base": base, "adjusted": adjusted, "breakdown": breakdown}


def handle_weather_compare(info: dict):
    normal_info = info.copy()
    rain_info = info.copy()
    normal_info["precip"] = 0.0
    normal_info["windgust"] = 20.0
    rain_info["precip"] = 12.0
    rain_info["windgust"] = 45.0

    _, normal_meta, normal_base, normal_adj, normal_breakdown = run_prediction_from_info(normal_info)
    _, rain_meta, rain_base, rain_adj, rain_breakdown = run_prediction_from_info(rain_info)
    increase = rain_adj - normal_adj

    df = pd.DataFrame([
        {"condition": "Normal", "final_fare": round(normal_adj, 2), "duration_min": normal_meta["estimated_duration_min"], "weather_add": normal_breakdown["weather_add"]},
        {"condition": "Rain", "final_fare": round(rain_adj, 2), "duration_min": rain_meta["estimated_duration_min"], "weather_add": rain_breakdown["weather_add"]},
    ])

    reply = f"""
Rain vs normal weather{format_route_text(info)}

Normal weather fare: ${normal_adj:.2f}  
Rain-condition fare: ${rain_adj:.2f}  
Estimated increase: ${increase:.2f}

Why: rain increases weather pressure and can reduce estimated travel speed, increasing duration-sensitive fare behavior.
"""
    return reply, {"table_df": df}


def handle_airport_explain(info: dict):
    _, meta, base, adjusted, breakdown = run_prediction_from_info(info)
    reply = f"""
Airport trip explanation{format_route_text(info)}

Estimated final fare: ${adjusted:.2f}

Airport trips are usually more expensive in this app because:
- Airport routes are often longer.
- Airport trips include airport-related features/adjustments when JFK/LGA/Newark is detected.
- Airport trips can have higher demand pressure.

Airport adjustment in this estimate: ${breakdown['airport_add']:.2f}.
"""
    return reply, {"meta": meta, "base": base, "adjusted": adjusted, "breakdown": breakdown}


def handle_feature_importance():
    reply = """
What affects the fare most in this prototype

1. Traffic — strongest practical adjustment source in our simulation.  
2. Demand / rush hour — raises fare pressure during peak windows.  
3. Airport trips — adds airport-related pricing pressure when JFK/LGA/Newark is involved.  
4. Weather — adds a smaller penalty during rain/snow/bad-weather conditions.  
5. Distance and duration — core drivers of the base model fare.

This is why the system is more than a fare predictor: it explains which condition is pushing the fare up.
"""
    return reply, {}


def handle_route_analysis(info: dict):
    _, meta, base, adjusted, breakdown = run_prediction_from_info(info)
    congestion_label = "congested" if meta["traffic_intensity"] >= 0.5 else "moderate / normal"
    reply = f"""
Route analysis{format_route_text(info)}

Estimated duration: {meta['estimated_duration_min']} min  
Estimated speed: {meta['speed_mph']} mph  
Traffic intensity: {meta['traffic_intensity']}  
Congestion level: {congestion_label}

Estimated final fare: ${adjusted:.2f}
"""
    return reply, {"meta": meta, "base": base, "adjusted": adjusted, "breakdown": breakdown}


def handle_distance_sensitivity(info: dict):
    distances = [3, 6, 10]
    if info.get("trip_distance"):
        base_d = float(info["trip_distance"])
        distances = sorted(set([max(1, round(base_d * 0.75, 1)), round(base_d, 1), round(base_d * 1.25, 1)]))

    rows = []
    for d in distances:
        test_info = info.copy()
        test_info["trip_distance"] = d
        _, meta, base, adj, breakdown = run_prediction_from_info(test_info)
        rows.append({"distance_miles": d, "final_fare": round(adj, 2), "duration_min": meta["estimated_duration_min"], "speed_mph": meta["speed_mph"]})

    df = pd.DataFrame(rows)
    lines = "\n".join([f"- {r['distance_miles']} miles → ${r['final_fare']:.2f}" for _, r in df.iterrows()])
    reply = f"""
Distance sensitivity{format_route_text(info)}

{lines}

As distance increases, the base fare and estimated duration generally increase, so the final fare rises too.
"""
    return reply, {"table_df": df}


def split_compare_prompt(prompt: str):
    lower = prompt.lower()
    for sep in [" vs ", " versus "]:
        if sep in lower:
            idx = lower.find(sep)
            return prompt[:idx].strip(), prompt[idx + len(sep):].strip()
    return None, None


def handle_compare_trips(prompt: str):
    left, right = split_compare_prompt(prompt)
    if not left or not right:
        return "To compare trips, ask like: Manhattan to JFK at 5 PM vs Brooklyn to JFK at 5 PM", {}

    info1, missing1 = extract_trip_info(left)
    info2, missing2 = extract_trip_info(right)

    if info2.get("pickup_hour") is None and info1.get("pickup_hour") is not None:
        info2["pickup_hour"] = info1["pickup_hour"]
        missing2 = [m for m in missing2 if "pickup time" not in m]

    if missing1 or missing2:
        return "I need clearer trip details for comparison. Example: Manhattan to JFK at 5 PM vs Brooklyn to JFK at 5 PM", {}

    _, meta1, base1, adj1, breakdown1 = run_prediction_from_info(info1)
    _, meta2, base2, adj2, breakdown2 = run_prediction_from_info(info2)
    diff = abs(adj1 - adj2)
    cheaper = "Trip 1" if adj1 <= adj2 else "Trip 2"

    df = pd.DataFrame([
        {"trip": "Trip 1", "route": f"{info1.get('route_origin_label', 'Origin')} → {info1.get('route_destination_label', 'Destination')}", "final_fare": round(adj1, 2), "duration_min": meta1["estimated_duration_min"], "rush_hour": "Yes" if meta1["is_rush_hour"] else "No"},
        {"trip": "Trip 2", "route": f"{info2.get('route_origin_label', 'Origin')} → {info2.get('route_destination_label', 'Destination')}", "final_fare": round(adj2, 2), "duration_min": meta2["estimated_duration_min"], "rush_hour": "Yes" if meta2["is_rush_hour"] else "No"},
    ])

    reply = f"""
Trip comparison

Trip 1: ${adj1:.2f}  
Trip 2: ${adj2:.2f}

Cheaper option: {cheaper}  
Difference: ${diff:.2f}

The difference comes mainly from route distance, estimated duration, airport status, demand, and traffic conditions.
"""
    return reply, {"table_df": df}


def handle_advanced_intent(intent: str, prompt: str, info: dict):
    if intent == "service_availability":
        return handle_service_availability(prompt)
    if intent == "route_availability":
        return handle_route_query(prompt)
    if intent == "best_time":
        return handle_best_time(info)
    if intent == "worst_time":
        return handle_worst_time(info)
    if intent == "save_money":
        return handle_save_money(info)
    if intent == "time_savings":
        return handle_time_savings(prompt, info)
    if intent == "why_expensive":
        return handle_why_expensive(info)
    if intent == "weather_compare":
        return handle_weather_compare(info)
    if intent == "airport_explain":
        return handle_airport_explain(info)
    if intent == "feature_importance":
        return handle_feature_importance()
    if intent == "route_analysis":
        return handle_route_analysis(info)
    if intent == "distance_sensitivity":
        return handle_distance_sensitivity(info)
    if intent == "compare_trips":
        return handle_compare_trips(prompt)
    return None, {}

# ============================================================
# UI HELPERS
# ============================================================
def money(x):
    return f"${x:,.2f}"

def render_metric_card(label, value, help_text="", highlight=False):
    card_class = "metric-card final-fare-card" if highlight else "metric-card"
    st.markdown(f"""
        <div class="{card_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
    """, unsafe_allow_html=True)

def render_prediction_dashboard(meta, base_pred, adjusted_pred, breakdown, route_note=None):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Estimated fare", money(adjusted_pred), "final estimated taxi fare", highlight=True)
    with c2:
        render_metric_card("Base fare", money(base_pred), "model prediction")
    with c3:
        render_metric_card("Duration", f"{meta['estimated_duration_min']} min", f"speed {meta['speed_mph']} mph")
    with c4:
        total_adj = breakdown['demand_add'] + breakdown['traffic_add'] + breakdown['weather_add'] + breakdown['airport_add']
        render_metric_card("Price change", money(total_adj), "added policy adjustments")

    st.markdown("#### Fare details")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Demand", money(breakdown['demand_add']))
    b2.metric("Traffic", money(breakdown['traffic_add']))
    b3.metric("Weather", money(breakdown['weather_add']))
    b4.metric("Airport", money(breakdown['airport_add']))

    chips = [
        f"Rush hour: {'Yes' if meta['is_rush_hour'] else 'No'}",
        f"Bad weather: {'Yes' if meta['bad_weather_day'] else 'No'}",
        f"Airport trip: {'Yes' if meta['is_airport'] else 'No'}",
        f"Demand index: {meta['demand_index']}",
    ]
    st.markdown("".join([f"<span class='status-chip'>{c}</span>" for c in chips]), unsafe_allow_html=True)
    if route_note:
        st.markdown(f"<div class='soft-note'>{route_note}</div>", unsafe_allow_html=True)


def get_osrm_route(origin_lon, origin_lat, dest_lon, dest_lat):
    """Fetch an actual road-following driving route from OSRM.

    Returns (coordinates, distance_miles, duration_min) if available.
    Falls back to None if internet/API fails.
    Coordinates are in [lon, lat] format for pydeck PathLayer.
    """
    try:
        import requests

        url = (
            f"https://router.project-osrm.org/route/v1/driving/"
            f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
            f"?overview=full&geometries=geojson"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("code") != "Ok" or not data.get("routes"):
            return None

        route = data["routes"][0]
        coords = route["geometry"]["coordinates"]
        distance_miles = route["distance"] / 1609.34
        duration_min = route["duration"] / 60
        return coords, distance_miles, duration_min

    except Exception:
        return None

def render_route_map(meta):
    coords_ok = all(meta.get(k) is not None for k in [
        "origin_lat", "origin_lon",
        "destination_lat", "destination_lon"
    ])
    if not coords_ok:
        return

    import pydeck as pdk

    origin = [meta["origin_lon"], meta["origin_lat"]]
    destination = [meta["destination_lon"], meta["destination_lat"]]

    # Try to get an actual road-following car route from OSRM.
    # If OSRM is unavailable, fall back to the direct centroid-to-centroid line.
    route_result = get_osrm_route(
        meta["origin_lon"],
        meta["origin_lat"],
        meta["destination_lon"],
        meta["destination_lat"]
    )

    if route_result:
        route_coords, road_distance_miles, road_duration_min = route_result
        route_caption = (
            f"Road route estimate: {road_distance_miles:.2f} miles · "
            f"{road_duration_min:.1f} min"
        )
    else:
        route_coords = [origin, destination]
        route_caption = "Road route unavailable, showing direct zone-centroid fallback."

    points_df = pd.DataFrame([
        {
            "lat": meta["origin_lat"],
            "lon": meta["origin_lon"],
            "label": meta.get("route_origin_label", "Pickup"),
            "type": "Pickup"
        },
        {
            "lat": meta["destination_lat"],
            "lon": meta["destination_lon"],
            "label": meta.get("route_destination_label", "Dropoff"),
            "type": "Dropoff"
        },
    ])

    route_df = pd.DataFrame([
        {"path": route_coords}
    ])

    midpoint_lat = (meta["origin_lat"] + meta["destination_lat"]) / 2
    midpoint_lon = (meta["origin_lon"] + meta["destination_lon"]) / 2

    st.markdown("### Route preview")

    layer_route = pdk.Layer(
        "PathLayer",
        data=route_df,
        get_path="path",
        get_color=[250, 204, 21, 255],
        get_width=7,
        width_min_pixels=5,
        rounded=True,
        pickable=True,
    )

    layer_points = pdk.Layer(
        "ScatterplotLayer",
        data=points_df,
        get_position="[lon, lat]",
        get_radius=120,
        get_fill_color=[250, 204, 21, 255],
        get_line_color=[17, 24, 39, 255],
        line_width_min_pixels=2,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=midpoint_lat,
        longitude=midpoint_lon,
        zoom=10,
        pitch=0,
    )

    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        initial_view_state=view_state,
        layers=[layer_route, layer_points],
        tooltip={
            "html": "<b>{type}</b><br>{label}",
            "style": {"backgroundColor": "#111827", "color": "white"}
        },
    )

    st.pydeck_chart(deck, width='stretch')

    st.caption(
        f"Pickup: {points_df.iloc[0]['label']} · "
        f"Dropoff: {points_df.iloc[1]['label']} · "
        f"{route_caption}"
    )


def get_zone_options_df():
    """Build pickup/dropdown options from the full TLC zone lookup."""
    zones = load_zone_lookup().copy()
    if zones.empty:
        return zones
    zones = zones[~zones["LocationID"].isin([264, 265])].copy()
    zones["display"] = zones["Borough"].astype(str) + " · " + zones["Zone"].astype(str) + " (ID " + zones["LocationID"].astype(str) + ")"
    zones = zones.sort_values(["Borough", "Zone", "LocationID"]).reset_index(drop=True)
    return zones

ZONE_OPTIONS_DF = get_zone_options_df()


def location_payload_from_zone_id(location_id: int):
    """Return the same location payload format used by chat route parsing."""
    zones = load_zone_lookup()
    row = zones[zones["LocationID"] == int(location_id)]
    if row.empty:
        fallback = BOROUGH_DEFAULTS["manhattan"]
        return {**fallback, "borough": "manhattan", "granularity": "place"}

    r = row.iloc[0]
    borough = normalize_location_text(str(r["Borough"]))
    zone_label = str(r["Zone"])
    borough_default = BOROUGH_DEFAULTS.get(borough, BOROUGH_DEFAULTS["manhattan"])
    centroid = GEOMETRY_ZONE_CENTROIDS.get(int(location_id)) or CURATED_ZONE_CENTROIDS.get(int(location_id), {"lat": borough_default["lat"], "lon": borough_default["lon"]})
    return {
        "id": int(location_id),
        "label": zone_label,
        "borough": borough,
        "lat": centroid["lat"],
        "lon": centroid["lon"],
        "granularity": "place",
    }


def weather_values_from_condition(weather_condition: str):
    """Map simple user-facing weather choices to model weather features."""
    condition = (weather_condition or "Clear").lower()

    if condition == "rain":
        return {
            "precip": 12.0,
            "windgust": 45.0,
            "snow": 0.0,
            "snowdepth": 0.0,
            "temp": 15.0,
            "humidity": 75.0,
        }

    if condition == "snow":
        return {
            "precip": 0.0,
            "windgust": 35.0,
            "snow": 2.0,
            "snowdepth": 2.0,
            "temp": 1.0,
            "humidity": 70.0,
        }

    return {
        "precip": 0.0,
        "windgust": 20.0,
        "snow": 0.0,
        "snowdepth": 0.0,
        "temp": 15.0,
        "humidity": 50.0,
    }


def build_info_from_selected_zones(origin_id, destination_id, pickup_hour, pickup_weekday, passenger_count, weather_condition="Clear", manual_distance=None):
    """Build model-ready trip info from selected pickup/drop-off TLC zones."""
    origin = location_payload_from_zone_id(int(origin_id))
    destination = location_payload_from_zone_id(int(destination_id))

    estimated_miles, route_note = estimate_trip_distance_from_locations(origin, destination)
    trip_distance = float(manual_distance) if manual_distance is not None else estimated_miles

    airport_zone_ids = {1, 132, 138}
    is_airport_trip = int(origin["id"] in airport_zone_ids or destination["id"] in airport_zone_ids)

    weather_values = weather_values_from_condition(weather_condition)

    return {
        "pickup_hour": pickup_hour,
        "pickup_weekday": pickup_weekday,
        "trip_distance": trip_distance,
        "passenger_count": passenger_count,
        "RatecodeID": 2 if is_airport_trip else 1,
        "payment_type": 1,
        "PULocationID": int(origin["id"]),
        "DOLocationID": int(destination["id"]),
        "tip_amount": 2.0,
        "tolls_amount": 0.0,
        "congestion_surcharge": 2.5,
        "Airport_fee": 1.25 if is_airport_trip else 0.0,
        "precip": weather_values["precip"],
        "windgust": weather_values["windgust"],
        "snowdepth": weather_values["snowdepth"],
        "snow": weather_values["snow"],
        "temp": weather_values["temp"],
        "humidity": weather_values["humidity"],
        "pickup_date": "2024-01-15",
        "route_note": route_note,
        "route_origin_label": origin["label"],
        "route_destination_label": destination["label"],
        "origin_lat": origin.get("lat"),
        "origin_lon": origin.get("lon"),
        "destination_lat": destination.get("lat"),
        "destination_lon": destination.get("lon"),
    }


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### NYC Taxi Intelligence")
st.sidebar.caption("NYC Yellow Taxi fare estimates powered by route, demand, traffic, and weather signals.")

if st.sidebar.button("Reset Chat", width="stretch"):
    st.session_state.clear()

st.sidebar.markdown("---")
st.sidebar.markdown("#### Try these")
st.sidebar.code("I want to go from Manhattan to JFK around evening")
st.sidebar.code("Is service available to Newark?")
st.sidebar.code("Best time to travel from Manhattan to JFK")
st.sidebar.code("How can I make Manhattan to JFK cheaper?")
st.sidebar.code("Manhattan to JFK at 2 pm instead of 5 pm")
st.sidebar.code("Manhattan to JFK at 5 pm vs Brooklyn to JFK at 5 pm")
st.sidebar.code("What if it rains from Times Square to JFK at 2 pm?")
st.sidebar.code("Times Square to JFK at 2 pm in snow")

st.sidebar.markdown("---")
st.sidebar.markdown("#### Model snapshot")
st.sidebar.metric("Model", "XGBoost")
st.sidebar.metric("R²", "0.968")
st.sidebar.metric("MAE", "$1.70")
st.sidebar.metric("RMSE", "$3.81")

with st.sidebar.expander("System status", expanded=False):
    st.write(GEOMETRY_STATUS)
    if GEOMETRY_ZONE_CENTROIDS:
        st.success("Zone centroid distance is active.")
    else:
        st.warning("Using fallback centroid estimates.")
    if EXPECTED_COLUMNS:
        st.write(f"Detected model input columns: {len(EXPECTED_COLUMNS)}")
    else:
        st.write("Could not auto-detect exact model input columns.")
    if client is None:
        st.info("OpenAI key not found. Trip answers use the deterministic built-in response.")

# ============================================================
# TABS
# ============================================================
tab1, tab2 = st.tabs(["Fare Estimator", "Pricing Assistant"])

with tab1:
    left, right = st.columns([1, 1.15], gap="large")

    if ZONE_OPTIONS_DF.empty:
        st.error("Could not load TLC zone lookup. Put taxi_zone_lookup.csv or Pasted text(12).txt in the same folder.")
    else:
        display_to_id = dict(zip(ZONE_OPTIONS_DF["display"], ZONE_OPTIONS_DF["LocationID"]))
        id_to_display = dict(zip(ZONE_OPTIONS_DF["LocationID"], ZONE_OPTIONS_DF["display"]))

        default_pickup_id = 230 if 230 in id_to_display else int(ZONE_OPTIONS_DF.iloc[0]["LocationID"])
        default_dropoff_id = 65 if 65 in id_to_display else int(ZONE_OPTIONS_DF.iloc[min(1, len(ZONE_OPTIONS_DF)-1)]["LocationID"])

        with left:
            st.markdown("### Build a trip")
            st.caption("Choose pickup and drop-off TLC zones. The app will infer distance and use the correct pickup/drop-off IDs for the model.")

            pickup_display = st.selectbox(
                "Pickup zone",
                options=list(display_to_id.keys()),
                index=list(display_to_id.keys()).index(id_to_display[default_pickup_id]),
                help="This maps directly to PULocationID."
            )
            dropoff_display = st.selectbox(
                "Drop-off zone",
                options=list(display_to_id.keys()),
                index=list(display_to_id.keys()).index(id_to_display[default_dropoff_id]),
                help="This maps directly to DOLocationID."
            )

            pickup_id = int(display_to_id[pickup_display])
            dropoff_id = int(display_to_id[dropoff_display])

            pickup_hour = st.slider("Pickup hour", min_value=0, max_value=23, value=17)
            pickup_weekday = st.selectbox(
                "Pickup weekday",
                options=[0, 1, 2, 3, 4, 5, 6],
                format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
                index=2,
            )
            passenger_count = st.number_input("Passenger count", min_value=1, max_value=6, value=1, step=1)

            c_weather, c_override = st.columns(2)
            with c_weather:
                weather_condition = st.selectbox(
                    "Weather condition",
                    options=["Clear", "Rain", "Snow"],
                    index=0,
                    help="Cloudy is excluded because it usually does not directly affect taxi fares unless it involves rain or snow."
                )
            with c_override:
                override_distance = st.checkbox("Override distance")

            preview_info = build_info_from_selected_zones(
                pickup_id,
                dropoff_id,
                pickup_hour,
                pickup_weekday,
                passenger_count,
                weather_condition=weather_condition,
                manual_distance=None,
            )
            inferred_distance = float(preview_info["trip_distance"])

            manual_distance = None
            if override_distance:
                manual_distance = st.number_input(
                    "Manual trip distance (miles)",
                    min_value=0.1,
                    max_value=100.0,
                    value=float(inferred_distance),
                    step=0.1,
                    help="Use this only when you want to test a custom distance instead of the zone-centroid estimate."
                )

            st.markdown("</div>", unsafe_allow_html=True)

        manual_info = build_info_from_selected_zones(
            pickup_id,
            dropoff_id,
            pickup_hour,
            pickup_weekday,
            passenger_count,
            weather_condition=weather_condition,
            manual_distance=manual_distance,
        )

        try:
            input_df, meta, base_pred, adjusted_pred, breakdown = run_prediction_from_info(manual_info)

            with right:
                st.markdown("### Prediction output")
                st.markdown(
                    f"<div class='soft-note'><b>Route:</b> {manual_info['route_origin_label']} → {manual_info['route_destination_label']} · "
                    f"<b>Estimated distance:</b> {manual_info['trip_distance']:.2f} miles · "
                    f"<b>Pickup ID:</b> {manual_info['PULocationID']} · <b>Drop-off ID:</b> {manual_info['DOLocationID']}</div>",
                    unsafe_allow_html=True,
                )
                render_prediction_dashboard(meta, base_pred, adjusted_pred, breakdown, manual_info.get("route_note"))

                with st.expander("Model input used"):
                    st.dataframe(input_df, width="stretch")

            st.markdown("---")
            st.markdown(
                "<div style='width:100vw; position:relative; left:50%; transform:translateX(-50%);'>",
                unsafe_allow_html=True
            )
            render_route_map(meta)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            with right:
                st.error(f"Prediction failed. Error: {e}")
                st.info("Most likely cause: your saved .pkl expects a slightly different set of columns than the app is sending.")

with tab2:
    top_col, hint_col = st.columns([1.4, 1], gap="large")
    with top_col:
        st.markdown("### Chat with the Pricing Assistant")
        st.caption("Examples: 'Best time to travel from Manhattan to JFK', 'Manhattan to Brooklyn at 2 pm', or '8 miles to JFK at 6 pm in rain'.")
    with hint_col:
        st.markdown("<div class='soft-note'>Tip: If you only give boroughs, the app uses a central route estimate and explains the assumption.</div>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Ask me about NYC Yellow Taxi trips, fares, routes, or best travel times. Example: 'Best time to travel from Manhattan to JFK'."}
        ]

    if "pending_trip_info" not in st.session_state:
        st.session_state.pending_trip_info = None

    if "awaiting_trip_details" not in st.session_state:
        st.session_state.awaiting_trip_details = False

    if "pending_trip_mode" not in st.session_state:
        st.session_state.pending_trip_mode = "single"

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask an NYC Yellow Taxi question...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        reply = None
        latest_meta = None
        latest_breakdown = None
        latest_base = None
        latest_adjusted = None

        try:
            intent = detect_advanced_intent(prompt)

            if st.session_state.awaiting_trip_details:
                info, missing = merge_partial_info(st.session_state.pending_trip_info, prompt)
                pending_mode = st.session_state.pending_trip_mode

                if pending_mode and pending_mode not in {"single", "optimize"}:
                    missing = missing_for_intent(info, missing, pending_mode)
                    if missing:
                        reply = "I still need: " + ", ".join(missing)
                        st.session_state.pending_trip_info = info
                    else:
                        reply, artifacts = handle_advanced_intent(pending_mode, prompt, info)
                        if artifacts.get("time_df") is not None:
                            st.session_state["latest_time_optimization_df"] = artifacts["time_df"]
                        if artifacts.get("table_df") is not None:
                            st.session_state["latest_table_df"] = artifacts["table_df"]
                        if artifacts.get("meta") is not None:
                            latest_meta = artifacts["meta"]
                            latest_breakdown = artifacts.get("breakdown")
                            latest_base = artifacts.get("base")
                            latest_adjusted = artifacts.get("adjusted")
                        st.session_state.awaiting_trip_details = False
                        st.session_state.pending_trip_info = None
                        st.session_state.pending_trip_mode = "single"

                elif pending_mode == "optimize":
                    missing = [m for m in missing if "pickup time" not in m]
                    if missing:
                        reply = "I still need: " + ", ".join(missing)
                        st.session_state.pending_trip_info = info
                    else:
                        result_df, best_row, worst_row, savings = optimize_best_travel_time(info)
                        reply = build_time_optimization_reply(info, best_row, worst_row, savings)
                        st.session_state["latest_time_optimization_df"] = result_df
                        st.session_state.awaiting_trip_details = False
                        st.session_state.pending_trip_info = None
                        st.session_state.pending_trip_mode = "single"
                else:
                    if missing:
                        reply = "I still need: " + ", ".join(missing)
                        st.session_state.pending_trip_info = info
                    else:
                        input_df, meta, base_pred, adjusted_pred, breakdown = run_prediction_from_info(info)
                        reply = fallback_trip_response(base_pred, adjusted_pred, meta, breakdown, info.get("route_note"))
                        latest_meta, latest_breakdown, latest_base, latest_adjusted = meta, breakdown, base_pred, adjusted_pred
                        st.session_state.awaiting_trip_details = False
                        st.session_state.pending_trip_info = None
                        st.session_state.pending_trip_mode = "single"

            elif intent:
                artifacts = {}
                if intent in {"feature_importance", "compare_trips", "service_availability", "route_availability"}:
                    reply, artifacts = handle_advanced_intent(intent, prompt, {})
                else:
                    info, missing = extract_trip_info(prompt)
                    missing = missing_for_intent(info, missing, intent)
                    if missing:
                        reply = "Need more info: " + ", ".join(missing)
                        st.session_state.awaiting_trip_details = True
                        st.session_state.pending_trip_info = info
                        st.session_state.pending_trip_mode = intent
                    else:
                        reply, artifacts = handle_advanced_intent(intent, prompt, info)

                if artifacts.get("time_df") is not None:
                    st.session_state["latest_time_optimization_df"] = artifacts["time_df"]
                if artifacts.get("table_df") is not None:
                    st.session_state["latest_table_df"] = artifacts["table_df"]
                if artifacts.get("meta") is not None:
                    latest_meta = artifacts["meta"]
                    latest_breakdown = artifacts.get("breakdown")
                    latest_base = artifacts.get("base")
                    latest_adjusted = artifacts.get("adjusted")

            elif detect_trip_query(prompt):
                info, missing = extract_trip_info(prompt)
                if missing:
                    reply = "Need more info: " + ", ".join(missing)
                    st.session_state.awaiting_trip_details = True
                    st.session_state.pending_trip_info = info
                    st.session_state.pending_trip_mode = "single"
                else:
                    input_df, meta, base_pred, adjusted_pred, breakdown = run_prediction_from_info(info)
                    reply = fallback_trip_response(base_pred, adjusted_pred, meta, breakdown, info.get("route_note"))
                    latest_meta, latest_breakdown, latest_base, latest_adjusted = meta, breakdown, base_pred, adjusted_pred

            else:
                reply = "I’m here to help with NYC Yellow Taxi trip estimates. Try asking: 'Best time to travel from Manhattan to JFK'."

        except Exception as e:
            reply = f"Something went wrong: {e}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

        if "latest_time_optimization_df" in st.session_state:
            result_df = st.session_state.pop("latest_time_optimization_df")
            st.markdown("---")
            st.markdown("#### Fare by hour")
            st.line_chart(result_df.set_index("time_label")["final_fare"], width="stretch")
            with st.expander("View hourly fare comparison"):
                st.dataframe(result_df, width="stretch")

        if "latest_table_df" in st.session_state:
            table_df = st.session_state.pop("latest_table_df")
            st.markdown("---")
            st.markdown("#### Comparison table")
            st.dataframe(table_df, width="stretch")

        if latest_meta and latest_breakdown:
            st.markdown("---")
            render_prediction_dashboard(latest_meta, latest_base, latest_adjusted, latest_breakdown, latest_meta.get("route_note"))
            render_route_map(latest_meta)

