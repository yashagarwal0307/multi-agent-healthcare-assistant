# healthcare_workflow.py
# healthcare_workflow.py
"""
Multi-Agent Healthcare Assistant pipeline (plain Python agents).
- IngestionAgent (image + PDF upload handling, OCR if available, de-id)
- ImagingAgent (dummy rule-based classifier)
- TherapyAgent (OTC mapping, age/allergy checks, interactions)
- PharmacyMatchAgent (simple geo matching + inventory reserve)
- DoctorEscalationAgent (mock tele-consult roster)
- Coordinator (routes & final consolidation, triggers escalation)
All data under /data (CSV/JSON).
"""
import os, io, json, base64, time, math, random, csv
from typing import Dict, Any, List, Optional
from datetime import datetime
try:
    import fitz  # pymupdf
except Exception:
    fitz = None
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None
import pandas as pd

ROOT = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
UPLOADS_DIR = os.path.join(ROOT, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ---------- Helpers ----------
def now_ts():
    return datetime.utcnow().isoformat() + "Z"

def update_state(state: Dict[str, Any], new_bits: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(state)
    merged.update(new_bits)
    return merged

def save_b64_file(b64: str, filename_prefix: str) -> str:
    buf = base64.b64decode(b64)
    ts = int(time.time()*1000)
    ext = ".bin"
    if filename_prefix.lower().endswith(".png") or filename_prefix.lower().endswith(".jpg") or filename_prefix.lower().endswith(".jpeg"):
        ext = os.path.splitext(filename_prefix)[1] or ".png"
    path = os.path.join(UPLOADS_DIR, f"{filename_prefix}_{ts}{ext}")
    with open(path, "wb") as f:
        f.write(buf)
    return path

def extract_text_from_pdf(path: str) -> str:
    if fitz:
        try:
            doc = fitz.open(path)
            text = ""
            for p in doc:
                text += p.get_text() or ""
            return text
        except Exception as e:
            return f"[pdf_extract_error:{e}]"
    # fallback: pytesseract on pages if installed & PIL available
    if pytesseract and Image:
        try:
            # simplistic: try converting first page as image if possible
            im = Image.open(path)
            return pytesseract.image_to_string(im)
        except Exception:
            return "[pdf_extract_unavailable]"
    return "[pdf_extraction_unavailable]"

def deidentify_text(text: str) -> str:
    # simple regex-based redaction: emails, phones, long names
    import re
    text = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", "[REDACTED_EMAIL]", text, flags=re.I)
    text = re.sub(r"\b\d{10}\b", "[REDACTED_PHONE]", text)
    text = re.sub(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", lambda m: "[REDACTED_NAME]" if len(m.group(1))>2 else m.group(1), text)
    return text

def haversine_km(lat1, lon1, lat2, lon2):
    # return km
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dlambda = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

# ---------- Load Data (safe) ----------
def load_json_safe(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_csv_safe(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

PHARMACIES = load_json_safe("pharmacies.json")
INVENTORY_DF = load_csv_safe("inventory.csv")
DOCTORS_DF = load_csv_safe("doctors.csv")
MEDS_DF = load_csv_safe("meds.csv")
INTERACTIONS_DF = load_csv_safe("interactions.csv")
ZIP_DF = load_csv_safe("zipcodes.csv")


# ========== AGENTS ==========

# 1) Ingestion Agent
def ingestion_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts keys:
      - xray_b64 (base64 image) OR xray_path
      - pdf_b64 (base64 pdf) OR pdf_path OR pdf_text
      - patient (dict): age, allergies list, pincode (optional)
    Outputs:
      - xray_path (saved file), pdf_path (saved file), pdf_text (extracted & de-id), patient
    """
    log = state.get("_log", [])
    log.append((now_ts(), "ingestion.start", {}))

    patient = state.get("patient", {"age": 30, "allergies": []})
    out = {"patient": patient, "_log": log}

    # image
    xray_b64 = state.get("xray_b64")
    if xray_b64:
        try:
            path = save_b64_file(xray_b64, "xray.png")
            out["xray_path"] = path
            log.append((now_ts(), "ingestion.xray_saved", {"path": path}))
        except Exception as e:
            log.append((now_ts(), "ingestion.xray_error", {"error": str(e)}))

    elif state.get("xray_path"):
        out["xray_path"] = state.get("xray_path")

    # pdf
    pdf_b64 = state.get("pdf_b64")
    if pdf_b64:
        try:
            path = save_b64_file(pdf_b64, "report.pdf")
            out["pdf_path"] = path
            extracted = extract_text_from_pdf(path)
            out["pdf_text"] = deidentify_text(extracted)
            log.append((now_ts(), "ingestion.pdf_saved_extracted", {"path": path}))
        except Exception as e:
            log.append((now_ts(), "ingestion.pdf_error", {"error": str(e)}))
            out["pdf_text"] = state.get("pdf_text", "")
    else:
        # accept pre-extracted pdf_text
        txt = state.get("pdf_text", "")
        out["pdf_text"] = deidentify_text(txt) if txt else ""

    log.append((now_ts(), "ingestion.end", {}))
    out["_log"] = log
    return update_state(state, out)

# cnn model
def load_medical_cnn_model():
    """
    Load a pre-trained model - options for demo:
    """
    # Option A: Use TensorFlow Hub pre-trained models
    try:
        model = tf.keras.applications.DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        # Add custom classification head
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        predictions = tf.keras.layers.Dense(4, activation='softmax')(x)  # 4 conditions
        
        return tf.keras.Model(inputs=model.input, outputs=predictions)
        
    except Exception as e:
        # Fallback: return a mock model that behaves realistically
        return create_mock_cnn()
        
# 2) Imaging Agent (dummy)
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

def imaging_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Real CNN-based medical image analysis
    """
    log = state.get("_log", [])
    log.append((now_ts(), "imaging_cnn.start", {}))
    
    xray_path = state.get("xray_path")
    if not xray_path or not os.path.exists(xray_path):
        return imaging_agent(state)  # fallback to dummy
    
    try:
        # Load and preprocess image
        img = Image.open(xray_path).convert('RGB')
        img = img.resize((224, 224))  # Standard size for most CNNs
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get prediction (using a pre-trained model)
        model = load_medical_cnn_model()  # You'd implement this
        predictions = model.predict(img_array)
        
        # Map to conditions
        conditions = ["normal", "pneumonia", "covid_suspect", "other"]
        probs = {cond: float(pred) for cond, pred in zip(conditions, predictions[0])}
        
        # Determine severity based on confidence and condition
        top_condition = max(probs, key=probs.get)
        top_prob = probs[top_condition]
        
        if top_prob > 0.8:
            severity = "severe" if top_condition != "normal" else "none"
        elif top_prob > 0.6:
            severity = "moderate" if top_condition != "normal" else "none"
        else:
            severity = "mild" if top_condition != "normal" else "none"
            
        out = {
            "imaging_result": {
                "condition_probs": probs,
                "severity_hint": severity,
                "top_condition": top_condition,
                "confidence": top_prob
            },
            "_log": log + [(now_ts(), "imaging_cnn.end", {"real_cnn_used": True})]
        }
        
    except Exception as e:
        log.append((now_ts(), "imaging_cnn.error", {"error": str(e)}))
        # Fallback to dummy classifier
        out = imaging_agent(state)
    
    return update_state(state, out)
# 3) Therapy Agent
def therapy_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    log = state.get("_log", [])
    log.append((now_ts(), "therapy.start", {}))
    imaging = state.get("imaging_result", {})
    cond_probs = imaging.get("condition_probs", {})
    top = max(cond_probs, key=cond_probs.get) if cond_probs else "normal"
    conf = cond_probs.get(top, 1.0)
    # lookup meds (OTC only) in MEDS_DF
    otc_rows = MEDS_DF[MEDS_DF["indication"].str.lower() == top.lower()] if not MEDS_DF.empty else pd.DataFrame()
    otc_options = []
    for _, r in otc_rows.iterrows():
        otc_options.append({
            "sku": str(r.get("sku")),
            "drug_name": r.get("drug_name"),
            "dose": r.get("dose", ""),
            "freq": r.get("freq", "")
        })
    # fallback suggestions if none
    if not otc_options:
        if top == "pneumonia":
            otc_options = [{"sku":"OTC001","drug_name":"Paracetamol 500mg","dose":"500mg","freq":"q8h"}]
        elif top == "covid_suspect":
            otc_options = [{"sku":"OTC003","drug_name":"Oral Rehydration","dose":"--","freq":"as needed"}]
        else:
            otc_options = [{"sku":"OTC002","drug_name":"Cough Syrup","dose":"10ml","freq":"q8h"}]

    # age/allergy checks
    patient = state.get("patient", {})
    age = int(patient.get("age", 30))
    allergies = [a.lower() for a in patient.get("allergies", [])]

    red_flags = []
    # example red flags (extendable)
    notes = state.get("pdf_text","").lower()
    if "chest pain" in notes or "shortness of breath" in notes or imaging.get("severity_hint") == "severe":
        red_flags.append("Chest pain or severe respiratory distress — advise immediate care")

    # contraindications via meds.csv contra_allergy_keywords column (comma separated)
    for opt in otc_options:
        sku = opt.get("sku")
        row = MEDS_DF[MEDS_DF["sku"] == sku] if not MEDS_DF.empty else pd.DataFrame()
        if not row.empty:
            contra = str(row.iloc[0].get("contra_allergy_keywords","")).lower()
            for a in allergies:
                if a and a in contra:
                    opt.setdefault("warnings", []).append(f"Allergy match: {a}")

    # basic interaction checks: pairwise interactions from INTERACTIONS_DF
    interactions_found = []
    for i in range(len(otc_options)):
        for j in range(i+1, len(otc_options)):
            a = otc_options[i]["sku"]; b = otc_options[j]["sku"]
            df = INTERACTIONS_DF[((INTERACTIONS_DF["drug_a"]==a)&(INTERACTIONS_DF["drug_b"]==b))|((INTERACTIONS_DF["drug_a"]==b)&(INTERACTIONS_DF["drug_b"]==a))] if not INTERACTIONS_DF.empty else pd.DataFrame()
            if not df.empty:
                for _,r in df.iterrows():
                    interactions_found.append({"pair":[a,b],"level":r.get("level"),"note":r.get("note")})
    if interactions_found:
        red_flags.append("Drug interactions detected — review before ordering")

    out = {
        "therapy_plan": {"condition": top, "confidence": conf, "otc_options": otc_options, "red_flags": red_flags},
        "_log": log + [(now_ts(), "therapy.end", {"therapy_plan": {"condition": top}})]
    }
    return update_state(state, out)

# 4) Pharmacy Match Agent
def pharmacy_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    log = state.get("_log", [])
    log.append((now_ts(), "pharmacy.start", {}))
    # choose nearest pharmacy: use patient pincode -> zip lat/lon, else pick first
    p = state.get("patient", {})
    pincode = p.get("pincode")
    patient_lat = None; patient_lon = None
    if pincode and not ZIP_DF.empty:
        df = ZIP_DF[ZIP_DF["pincode"].astype(str) == str(pincode)]
        if not df.empty:
            patient_lat = float(df.iloc[0]["lat"]); patient_lon = float(df.iloc[0]["lon"])
    # fallback lat/lon if provided directly
    if not patient_lat:
        patient_lat = p.get("lat"); patient_lon = p.get("lon")

    # compute score (distance + stock coverage)
    therapy = state.get("therapy_plan", {})
    items = therapy.get("otc_options", [])
    requested_skus = [it.get("sku") for it in items]

    best = None
    best_score = -1e9
    for ph in PHARMACIES:
        plat = ph.get("lat"); plon = ph.get("lon")
        dist_km = 9999
        if patient_lat and plat:
            dist_km = haversine_km(patient_lat, patient_lon, float(plat), float(plon))
        # count available qty from INVENTORY_DF
        count = 0
        for sku in requested_skus:
            if INVENTORY_DF.empty:
                # assume available
                count += 1
            else:
                row = INVENTORY_DF[(INVENTORY_DF["pharmacy_id"]==ph.get("id")) & (INVENTORY_DF["sku"]==sku)]
                if not row.empty and int(row.iloc[0].get("qty",0))>0:
                    count += 1
        # scoring heuristic: prefer more items and nearer
        score = count*100 - dist_km
        if score > best_score:
            best_score = score; best = ph; best_dist = dist_km; best_count = count

    if best is None:
        match = {"pharmacy_id": None, "items": [], "eta_min": None, "delivery_fee": None}
    else:
        eta = int(best.get("delivery_km", 30))  # mock mapping
        fee = int(best.get("delivery_fee", 25)) if best.get("delivery_fee") else 25
        match_items = []
        for sku in requested_skus:
            # quantity selection: available qty in inventory or default 10
            qty = 10
            if not INVENTORY_DF.empty:
                row = INVENTORY_DF[(INVENTORY_DF["pharmacy_id"]==best.get("id")) & (INVENTORY_DF["sku"]==sku)]
                if not row.empty:
                    qty = int(min(10, row.iloc[0].get("qty", 0)))
            match_items.append({"sku": sku, "qty": qty})
        match = {"pharmacy_id": best.get("id"), "name": best.get("name"), "items": match_items, "eta_min": eta, "delivery_fee": fee}
    out = {"pharmacy_match": match, "_log": log + [(now_ts(), "pharmacy.end", {"match": match})]}
    return update_state(state, out)

# 5) Doctor Escalation Agent
def doctor_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    log = state.get("_log", [])
    log.append((now_ts(), "doctor.start", {}))
    # choose a doctor from DOCTORS_DF or mock
    if not DOCTORS_DF.empty:
        row = DOCTORS_DF.sample(1).iloc[0]
        scheduled = row.get("tele_slot_iso8601", "").split(";")[0] if row.get("tele_slot_iso8601") else None
        doc = {"doctor_id": row.get("doctor_id"), "name": row.get("name"), "specialty": row.get("specialty"), "tele_slot": scheduled}
    else:
        doc = {"doctor_id": "doc001", "name": "Dr. Smith", "specialty": "Pulmonology", "tele_slot": "2025-10-05T16:00:00Z"}
    out = {"doctor_escalation": doc, "_log": log + [(now_ts(), "doctor.end", {"doctor": doc})]}
    return update_state(state, out)

# 6) Coordinator / Orchestrator
CONF_THRESHOLD = 0.40
def coordinator(state: Dict[str, Any]) -> Dict[str, Any]:
    log = state.get("_log", [])
    log.append((now_ts(), "coordinator.start", {}))
    therapy = state.get("therapy_plan", {})
    red_flags = therapy.get("red_flags", [])
    conf = therapy.get("confidence", 1.0)
    # escalate if red flags or low confidence
    if red_flags or conf < CONF_THRESHOLD:
        log.append((now_ts(), "coordinator.escalation_decision", {"conf": conf, "red_flags": red_flags}))
        state = doctor_agent(state)
    else:
        log.append((now_ts(), "coordinator.no_escalation", {"conf": conf}))
    # create final output JSON
    final = {
        "patient": state.get("patient"),
        "condition": therapy.get("condition"),
        "confidence": therapy.get("confidence"),
        "otc_options": therapy.get("otc_options"),
        "red_flags": therapy.get("red_flags"),
        "pharmacy": state.get("pharmacy_match"),
        "doctor": state.get("doctor_escalation"),
        "disclaimer": "Educational demo only — Not medical advice.",
        "timestamp": now_ts()
    }
    log.append((now_ts(), "coordinator.end", {"final_summary_present": True}))
    return update_state(state, {"final_output": final, "_log": log})

# ---------- Pipeline runner ----------
def run_pipeline(input_state: Dict[str, Any]) -> Dict[str, Any]:
    # pipeline: ingestion -> imaging -> therapy -> pharmacy -> coordinator
    state = dict(input_state)
    state.setdefault("_log", [])
    state = ingestion_agent(state)
    state = imaging_agent(state)
    state = therapy_agent(state)
    state = pharmacy_agent(state)
    state = coordinator(state)
    return state

# If run as script for quick test
import io, sys, contextlib

if __name__ == "__main__":
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    random.seed(42)
    demo_state = {
        "patient": {"age":45,"allergies":["ibuprofen"], "pincode": "400053"},
        "pdf_text": "Patient complains of cough and mild fever. No chest pain.",
        "_seed": 123
    }

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):  # silence TF prints
        out = run_pipeline(demo_state)

    print(json.dumps(out.get("final_output", {}), indent=2))
