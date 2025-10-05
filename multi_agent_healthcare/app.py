# app.py
import streamlit as st
import base64, json, os
from healthcare_workflow import run_pipeline
from healthcare_workflow import UPLOADS_DIR
st.set_page_config(page_title="Multi-Agent Healthcare Assistant", layout="wide")
st.title("🩺 Multi-Agent Healthcare Assistant — Demo")
st.markdown("**Disclaimer:** Educational demo only — Not medical advice.")

with st.sidebar:
    st.header("Upload patient data")
    uploaded_xray = st.file_uploader("Chest X-ray (PNG/JPG)", type=["png","jpg","jpeg"])
    uploaded_pdf = st.file_uploader("Optional Report (PDF)", type=["pdf"])
    st.markdown("---")
    st.header("Patient meta")
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    gender = st.selectbox("Gender", ["unknown","M","F","Other"])
    allergies = st.text_input("Allergies (comma-separated)", value="ibuprofen")
    pincode = st.text_input("Pincode (optional)", value="")

run = st.button("Run Pipeline")

def to_b64(file):
    return base64.b64encode(file.read()).decode("utf-8")

if run:
    if not uploaded_xray:
        st.error("Please upload a chest X-ray image to run the demo.")
    else:
        st.info("Running agents — this runs locally in the app process.")
        payload = {
            "patient": {"age": int(age), "gender": gender, "allergies":[a.strip() for a in allergies.split(",") if a.strip()], "pincode": pincode}
        }
        payload["xray_b64"] = to_b64(uploaded_xray)
        if uploaded_pdf:
            payload["pdf_b64"] = to_b64(uploaded_pdf)
        # run pipeline
        try:
            result_state = run_pipeline(payload)
            final = result_state.get("final_output", {})
            st.success("Pipeline complete")
            st.subheader("Final Summary")
            st.json(final)
            st.subheader("Event Log (chronological)")
            for ts,event,p in result_state.get("_log",[]):
                st.markdown(f"**{ts}** — `{event}` — {p}")
            # show saved image
            saved = result_state.get("xray_path")
            if saved and os.path.exists(saved):
                st.image(saved, caption="Saved X-ray (demo copy)", use_column_width=True)
            # allow download of order JSON
            st.download_button("Download order JSON", data=json.dumps(final, indent=2), file_name="order.json", mime="application/json")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
