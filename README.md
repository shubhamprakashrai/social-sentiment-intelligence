# 📊 Social Sentiment Analysis & Brand Intelligence Engine
> Aspect-Based Sentiment Analysis (ABSA) & Early Crisis Anomaly Detection microservice for D2C consumer brands.

---

## 🎯 Business Context (The Problem)
* **Company:** W3villa Technologies Pvt. Ltd. (Client: Indian D2C Cosmetics Brand)
* **Challenge:** High volume of noisy customer feedback across Amazon, Nykaa, and Twitter. Traditional sentiment analysis yields single scalar scores (e.g., "60% Negative"), failing to distinguish between product quality defects (skin allergies) and operational friction (delayed courier or cracked cap).
* **Solution:** A RoBERTa-inspired Aspect-Based Sentiment Analysis (ABSA) engine that decouples reviews across 5 business dimensions: `SKIN_REACTION`, `PACKAGING`, `FRAGRANCE`, `PRICE_VALUE`, and `DELIVERY`. Couples with a velocity anomaly detector to trigger automated product quarantine alerts before viral social crises escalate.

---

## 🏗️ Architecture Overview

```
[ Unstructured Reviews Ingestion (Amazon, Nykaa, Twitter) ]
                            │
                            ▼
[ Aspect Extraction & Sentiment Decoupling (ABSA) ]
  • Decouples mixed sentiments: "Great glow (+), but broken dropper (-)"
  • Flags clinical safety hazards (e.g., burning sensation, rashes)
                            │
                            ▼
[ Early Crisis Anomaly Spike Detector ]
  • Aggregates negative velocity per aspect
  • Triggers emergency quality audits when complaint threshold >= 2
                            │
                            ▼
[ REST Microservice API (FastAPI) ]
  • POST /analyze-single
  • POST /analyze-batch-and-detect-spikes
```

---

## 🚀 Quick Start & Testing

```bash
# 1. Run automated test suite
$env:PYTHONPATH="."; python tests/test_sentiment.py

# 2. Launch FastAPI Server
$env:PYTHONPATH="."; python -m uvicorn src.app:app --port 8002 --reload
```
