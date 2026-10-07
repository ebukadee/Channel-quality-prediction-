# Intelligent Channel Quality Prediction

A non-complex machine-learning mini project built with Python, Streamlit and scikit-learn.

## Features
- Futuristic minimalist dashboard
- SNR, RSSI, interference, latency and bandwidth inputs
- Random Forest channel-quality prediction (0–100)
- EXCELLENT / FAIR / POOR classification
- Automatic recommendation engine
- Model performance indicators
- No external database or API required

## Project structure
```text
intelligent_channel_quality_prediction/
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
└── sample_input.csv
```

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit, normally `http://localhost:8501`.

## Deployment
The easiest public deployment is Streamlit Community Cloud:
1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and the other project files.
3. Sign in to Streamlit Community Cloud with GitHub.
4. Create a new app.
5. Select the repository, branch, and `app.py`.
6. Deploy.

The included requirements file ensures the cloud environment installs the exact Python packages used by the project.
