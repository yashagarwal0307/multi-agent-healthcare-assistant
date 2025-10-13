# app.py
import streamlit as st
import base64, json, os, re
from healthcare_workflow import run_pipeline, ingestion_agent, UPLOADS_DIR

# ------------------------------------------------------------
# Streamlit UI setup
# ------------------------------------------------------------
st.set_page_config(page_title="Multi-Agent Healthcare Assistant", layout="wide")
st.title("🩺 Multi-Agent Healthcare Assistant — Demo")
st.markdown("**Disclaimer:** Educational demo only — Not medical advice.")
st.markdown("""
This is a simulation-only healthcare assistant built with modular agents.  
It accepts chest X-rays and optional reports, and suggests OTC treatments.  
All recommendations are strictly non-clinical and for educational purposes only.
""")

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def to_b64(file):
    return base64.b64encode(file.read()).decode("utf-8")

def extract_patient_metadata(text):
    """Simple regex-based metadata extractor for age, gender, and allergies."""
    metadata = {}
    age_match = re.search(r'Age[:\s]+(\d{1,2})', text)
    if age_match:
        metadata["age"] = int(age_match.group(1))

    gender_match = re.search(r'Gender[:\s]+(Male|Female|Other|M|F)', text, re.IGNORECASE)
    if gender_match:
        val = gender_match.group(1).lower()
        metadata["gender"] = "M" if val.startswith("m") else "F" if val.startswith("f") else "Other"

    allergies_match = re.findall(r'Allerg(?:y|ies)[:\s]+([A-Za-z,\s]+)', text)
    if allergies_match:
        metadata["allergies"] = [a.strip() for a in allergies_match[0].split(",") if a.strip()]

    return metadata

# ------------------------------------------------------------
# Sidebar UI for inputs
# ------------------------------------------------------------
with st.sidebar:
    st.header("Upload patient data")
    uploaded_xray = st.file_uploader("Chest X-ray (PNG/JPG)", type=["png","jpg","jpeg"])
    uploaded_pdf = st.file_uploader("Optional Report (PDF)", type=["pdf"])

    extracted_patient = {}  # will be populated if pdf is uploaded

    # If PDF uploaded, run ingestion agent
    if uploaded_pdf is not None:
        temp_path = os.path.join(UPLOADS_DIR, "temp_uploaded.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_pdf.read())

        try:
            # Run ingestion agent to extract text
            result = ingestion_agent({"pdf_path": temp_path})
            pdf_text = result.get("pdf_text", "")
            extracted_patient = extract_patient_metadata(pdf_text)
            st.success("✅ Patient info auto-extracted from PDF.")
        except Exception as e:
            st.warning(f"⚠️ Could not extract patient info: {e}")

    # Default fallback values
    age_val = extracted_patient.get("age", 45)
    gender_val = extracted_patient.get("gender", "unknown")
    allergy_val = ", ".join(extracted_patient.get("allergies", ["ibuprofen"]))

    st.markdown("---")
    st.header("Patient meta")

    age = st.number_input("Age", min_value=0, max_value=120, value=age_val)
    gender = st.selectbox(
        "Gender",
        ["unknown","M","F","Other"],
        index=["unknown","M","F","Other"].index(gender_val) if gender_val in ["unknown","M","F","Other"] else 0
    )
    allergies = st.text_input("Allergies (comma-separated)", value=allergy_val)
    pincode = st.text_input("Pincode (optional)", value="")

# ------------------------------------------------------------
# Main button to run pipeline
# ------------------------------------------------------------
run = st.button("Run Pipeline")

if run:
    if not uploaded_xray:
        st.error("Please upload a chest X-ray image to run the demo.")
    else:
        st.info("Running agents — this runs locally in the app process.")

        payload = {
            "patient": {
                "age": int(age),
                "gender": gender,
                "allergies": [a.strip() for a in allergies.split(",") if a.strip()],
                "pincode": pincode
            }
        }

        payload["xray_b64"] = to_b64(uploaded_xray)
        if uploaded_pdf:
            # re-encode PDF for pipeline
            uploaded_pdf.seek(0)
            payload["pdf_b64"] = to_b64(uploaded_pdf)

        # Run the multi-agent pipeline
        try:
            result_state = run_pipeline(payload)
            final = result_state.get("final_output", {})

            st.success("✅ Pipeline complete")
            st.subheader("Final Summary")
            st.json(final)

            st.subheader("Event Log (chronological)")
            for ts, event, p in result_state.get("_log", []):
                st.markdown(f"**{ts}** — `{event}` — {p}")

            # Show saved X-ray
            saved = result_state.get("xray_path")
            if saved and os.path.exists(saved):
                st.image(saved, caption="Saved X-ray (demo copy)", use_column_width=True)

            # Allow order JSON download
            st.download_button(
                "Download order JSON",
                data=json.dumps(final, indent=2),
                file_name="order.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"Pipeline error: {e}")
