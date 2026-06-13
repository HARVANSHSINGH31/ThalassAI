# ThalassAI

**An ML-based multi-signal ocean ecosystem stress detection system for the Arabian Sea.**

🚀 **Live Demo:** https://oceanpulse.streamlit.app

---

## What it does

OceanPulse fuses three oceanographic signals — sea surface temperature (SST), dissolved oxygen, and chlorophyll-a — into a composite stress index for the Arabian Sea, then applies an LSTM Autoencoder to detect anomalous stress events without labelled training data.

The system trained on 2015–2023 monthly data from Copernicus Marine Service (CMEMS) and, with no prior knowledge of documented marine events, correctly flagged all four major Arabian Sea stress periods recorded in literature:

| Event | Period | Signal drivers |
|---|---|---|
| Hypoxic stress event | Jul–Aug 2015 | DO collapse, SST spike |
| Major dead zone expansion | Mar–Jul 2018 | DO + Chl-a anomaly |
| Intensified hypoxia | Jul 2019 | DO minimum, elevated SST |
| Compounded stress | Jul–Aug 2020 | Multi-signal deviation |

---

## Data sources

All signals sourced from [Copernicus Marine Environment Monitoring Service (CMEMS)](https://marine.copernicus.eu/):

| Signal | Product | Resolution |
|---|---|---|
| Sea Surface Temperature | GLORYS12V1 physical reanalysis | Monthly, 1/12° |
| Dissolved Oxygen | GLOBAL_MULTIYEAR_BGC_001_029 | Monthly |
| Chlorophyll-a | GLOBAL_MULTIYEAR_BGC_001_029 | Monthly |

**Coverage:** 108 monthly observations, January 2015 – December 2023  
**Region:** Arabian Sea (15°N–25°N, 55°E–70°E)

---

## Architecture

**Model:** PyTorch LSTM Autoencoder  
**Anomaly criterion:** Reconstruction MSE exceeding 2 standard deviations above training mean  
**Composite stress score:** Weighted fusion — SST (40%), DO (35%), Chl-a (25%)

---

## Stack

Python · PyTorch · Pandas · NumPy · Scikit-learn · Plotly · Streamlit · Copernicus Marine API

---

## Run locally

```bash
git clone https://github.com/HARVANSHSINGH31/oceanpulse
cd oceanpulse
pip install -r requirements.txt
streamlit run app.py
```

To re-fetch CMEMS data, set your credentials in `.env`:

---

## Key results

The LSTM correctly identified all four documented Arabian Sea hypoxic and thermal stress events purely from reconstruction error — no event labels were used in training. This validates the unsupervised anomaly detection approach against independent oceanographic literature.

---

## Author

Harvansh Singh — (AI/ML), Independent ML Research
GitHub: [@HARVANSHSINGH31](https://github.com/HARVANSHSINGH31)
