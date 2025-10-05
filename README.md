# Multi-Agent Healthcare Assistant — Demo

Educational demo implementing a multi-agent pipeline for simulated COVID-style triage.
**Not medical advice.** This is a proof-of-concept for assignment/demonstration only.

## What it does
1. Upload chest X-ray (PNG/JPG) + optional PDF report/ID.
2. System triages likely condition(s) & severity.
3. Suggests non-prescriptive OTC options with warnings & interaction checks.
4. Matches to nearest partner pharmacy with stock; computes ETA & fee.
5. Offers optional mock tele-consult escalation if red-flag/low confidence.
6. Places a mock order & returns confirmation JSON.

## Tech
- Backend logic: `healthcare_workflow.py` (plain Python agents)
- UI: Streamlit (`app.py`)
- Dummy data: `data/` folder (pharmacies, inventory, meds, interactions, doctors, zipcodes)
- Tests: `tests/test_agents.py` (pytest)

## Quick start (local)
```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-healthcare.git
cd multi_agent_healthcare
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
