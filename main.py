from pathlib import Path
from typing import List
import joblib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "crop_model.joblib"

app = FastAPI(title="AgriSense AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://frontend-brown-three-43.vercel.app",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bundle = None
if MODEL_PATH.exists():
    bundle = joblib.load(MODEL_PATH)

class CropInput(BaseModel):
    N: float = Field(ge=0, le=300)
    P: float = Field(ge=0, le=300)
    K: float = Field(ge=0, le=300)
    temperature: float = Field(ge=-10, le=60)
    humidity: float = Field(ge=0, le=100)
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0, le=2000)

@app.get("/")
def root():
    return {"message": "AgriSense AI API is running", "docs": "/docs"}

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}

@app.get("/api/metrics")
def metrics():
    import json
    path = BASE / "models" / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Train the model first with: python train.py")
    return json.loads(path.read_text())

@app.post("/api/recommend")
def recommend(data: CropInput):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not found. Run python train.py after adding the dataset.")
    model = bundle["model"]
    features = bundle["features"]
    row = [[getattr(data, f) for f in features]]
    probabilities = model.predict_proba(row)[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3]
    return {
        "recommendations": [
            {"crop": str(crop), "score": round(float(score), 4)}
            for crop, score in ranked
        ],
        "note": "Scores are model probability estimates, not guaranteed real-world suitability percentages."
    }

@app.get("/api/weather")
def weather(city: str = "Hyderabad"):
    """Fetch current forecast context by city; weather is displayed separately from seasonal rainfall input."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geo.raise_for_status()
        results = geo.json().get("results", [])
        if not results:
            raise HTTPException(status_code=404, detail=f"Location not found: {city}")
        place = results[0]
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,precipitation",
                "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
                "forecast_days": 3,
                "timezone": "auto",
            },
            timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        return {
            "location": f'{place.get("name", city)}, {place.get("admin1", "")}, {place.get("country", "")}'.strip(", "),
            "current": body.get("current", {}),
            "daily": body.get("daily", {}),
            "source": "Open-Meteo",
            "note": "Forecast weather is contextual information; daily/short-term rainfall is not the same as seasonal rainfall used by the crop model."
        }
    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Weather service unavailable: {e}")

@app.get("/api/crops/{crop_name}")
def crop_info(crop_name: str):
    # Lightweight educational notes; expand using verified regional agriculture sources.
    tips = {
        "rice": "Typically needs substantial water and is often grown in wet or irrigated conditions. Check local season and water availability.",
        "maize": "Needs suitable drainage and adequate moisture; timing depends on local climate and season.",
        "cotton": "Requires a suitable warm growing season and locally appropriate soil and water conditions.",
        "chickpea": "Often grown as a cool-season pulse; check local sowing calendar and soil drainage.",
        "mungbean": "A short-duration pulse in many regions; local variety and season matter.",
    }
    return {
        "crop": crop_name,
        "general_note": tips.get(crop_name.lower(), "Check locally validated crop calendars, water availability, variety requirements, and advice from an agricultural extension officer."),
        "disclaimer": "General educational information only; not a substitute for local agronomic advice."
    }


class HybridInput(BaseModel):
    location: str = Field(min_length=2, max_length=120)
    state: str = Field(default="Telangana", min_length=2, max_length=80)
    season: str = Field(pattern="^(Kharif|Rabi|Summer)$")
    irrigation: str = Field(pattern="^(Rain-fed|Irrigation|Both)$")
    has_soil_report: bool = False

@app.post("/api/hybrid-recommend")
def hybrid_recommend(data: HybridInput):
    """
    Early prototype: returns season/region-based candidate crops.
    The candidate lists are illustrative and must be validated against
    current state agriculture department recommendations before real use.
    It intentionally does not invent NPK/pH values or claim precise suitability.
    """
    state = data.state.strip().lower()
    season = data.season
    irrigation = data.irrigation

    regional = {
        "telangana": {
            "Kharif": [
                ("Rice", "Commonly grown in the monsoon season where water and suitable fields are available.", "Check irrigation, local variety, and official sowing window."),
                ("Cotton", "A regional Kharif crop in suitable areas.", "Check soil drainage, rainfall outlook, and local pest guidance."),
                ("Maize", "A possible seasonal option in suitable fields.", "Check local rainfall, soil, and sowing calendar."),
                ("Red gram (pigeon pea)", "A pulse crop grown in parts of the region.", "Check soil drainage, duration, and local variety.")
            ],
            "Rabi": [
                ("Chickpea", "A possible cool-season pulse option in suitable areas.", "Check residual moisture, soil type, and district crop advisories."),
                ("Maize", "Can be considered where season and water supply are suitable.", "Check water availability and local sowing guidance."),
                ("Sorghum (jowar)", "A possible dryland crop option in suitable areas.", "Check local rainfall, soil, and official crop calendar.")
            ],
            "Summer": [
                ("Mung bean (green gram)", "A short-duration pulse that may fit some irrigated summer systems.", "Check local temperature, water availability, and sowing window."),
                ("Maize", "May be considered where irrigation and local conditions permit.", "Check heat, water supply, and district guidance.")
            ]
        }
    }
    crop_rows = regional.get(state, {}).get(season)
    if not crop_rows:
        # Safe fallback: no claim of state-specific validation.
        crop_rows = [
            ("Maize", "General candidate for further local suitability checking.", "Confirm local season, soil, water, and state crop advisories."),
            ("Pulses (locally suitable variety)", "A broad crop group to discuss with a local agriculture extension officer.", "Choose a specific pulse only after checking district recommendations."),
            ("Millets (locally suitable variety)", "A broad crop group that may suit some rain-fed systems.", "Confirm soil, rainfall, season, and local variety.")
        ]

    if irrigation == "Rain-fed":
        crop_rows = [r for r in crop_rows if r[0].lower() not in ("rice", "mung bean (green gram)")]
        if not crop_rows:
            crop_rows = regional.get(state, {}).get(season, [])[:2]

    candidates = [
        {"crop": name, "reason": reason, "check_before_planting": check}
        for name, reason, check in crop_rows[:3]
    ]
    return {
        "location": data.location,
        "state": data.state,
        "season": season,
        "irrigation": irrigation,
        "recommendation_type": "preliminary regional shortlist",
        "candidates": candidates,
        "soil_report_provided": data.has_soil_report,
        "notice": (
            "This prototype shortlist is not a validated farm-specific prediction. "
            "It does not infer NPK or pH from location. Confirm choices using current "
            "district crop advisories, soil testing, water availability, and an agricultural extension officer."
        ),
        "data_note": (
            "State/season candidate rules in this demo are illustrative. "
            "They must be replaced or verified with current official regional crop calendars before real-world use."
        )
    }
