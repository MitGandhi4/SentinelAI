import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.enrichment.enrichment import load_attack_data
from src.agent.agent import analyze_alert
from src.reporting.report_generator import generate_incident_report, format_report_as_markdown

# ── 1. Initialize FastAPI app ──────────────────────────────────────
app = FastAPI(
    title="SentinelAI API",
    description="Autonomous SecOps agent API for threat detection and analysis",
    version="1.0.0"
)

# ── 2. Load shared resources once at startup ───────────────────────
print("Loading ATT&CK data at API startup...")
TECHNIQUES = load_attack_data("data/raw/enterprise-attack.json")

# ── 3. Define request/response schemas ─────────────────────────────
class FlowFeatures(BaseModel):
    flow_duration: float
    total_fwd_packets: float
    total_bwd_packets: float
    flow_bytes_per_sec: float
    destination_port: int

class AnalyzeRequest(BaseModel):
    attack_label: str
    anomaly_score: float
    flow_features: FlowFeatures

class ReportRequest(BaseModel):
    attack_label: str
    anomaly_score: float

class HealthResponse(BaseModel):
    status: str
    techniques_loaded: int

# ── 4. Health check endpoint ───────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Simple endpoint to verify the API is running and data is loaded.
    """
    return HealthResponse(
        status="healthy",
        techniques_loaded=len(TECHNIQUES)
    )


# ── 5. Agent analysis endpoint ─────────────────────────────────────
@app.post("/analyze")
def analyze_endpoint(request: AnalyzeRequest):
    """
    Runs the LangGraph agent on a detected alert and returns
    a free-form security analysis.
    """
    try:
        analysis = analyze_alert(
            attack_label=request.attack_label,
            anomaly_score=request.anomaly_score,
            flow_features=request.flow_features.dict()
        )
        return {
            "attack_label": request.attack_label,
            "anomaly_score": request.anomaly_score,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 6. Incident report generation endpoint ────────────────────────
@app.post("/report")
def report_endpoint(request: ReportRequest):
    """
    Generates a structured incident report with both JSON and
    Markdown formats.
    """
    try:
        report = generate_incident_report(
            attack_label=request.attack_label,
            anomaly_score=request.anomaly_score
        )
        markdown = format_report_as_markdown(report)
        return {
            "report_json": report,
            "report_markdown": markdown
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 7. Root endpoint ────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "SentinelAI API is running",
        "docs": "/docs"
    }