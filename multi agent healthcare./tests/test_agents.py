# tests/test_agents.py
import json, random
from healthcare_workflow import ingestion_agent, imaging_agent, therapy_agent, pharmacy_agent, coordinator, run_pipeline

def test_ingestion_basic():
    payload = {"patient":{"age":30,"allergies":[]}, "pdf_text":"No acute issues", "_seed":42}
    state = ingestion_agent(payload)
    assert "patient" in state
    assert "_log" in state

def test_imaging_then_therapy():
    # fixed seed for deterministic-ish outcome
    payload = {"patient":{"age":40,"allergies":[]},"_seed":7}
    state = ingestion_agent(payload)
    state = imaging_agent(state)
    assert "imaging_result" in state
    state = therapy_agent(state)
    assert "therapy_plan" in state
    assert isinstance(state["therapy_plan"].get("otc_options"), list)

def test_full_pipeline_flow():
    random.seed(123)
    payload = {"patient":{"age":65,"allergies":["aspirin"], "pincode":"400053"}, "pdf_text":"Severe shortness of breath"}
    final_state = run_pipeline(payload)
    assert "final_output" in final_state
    out = final_state["final_output"]
    assert "disclaimer" in out
    assert out["patient"]["age"] == 65
