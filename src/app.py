from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.aspect_classifier import AspectBasedSentimentClassifier, ReviewAnalysisResult
from src.trend_detector import TrendAnomalyDetector, AnomalyAlert

app = FastAPI(
    title="Social Sentiment & Brand Intelligence Engine",
    description="Aspect-based sentiment analysis and product quality anomaly detection microservice.",
    version="1.0.0"
)

classifier = AspectBasedSentimentClassifier()
detector = TrendAnomalyDetector(spike_threshold=2)

class ReviewInput(BaseModel):
    review_id: str = Field(default="REV_LIVE_01")
    product_name: str = Field(default="GlowBoost Vitamin C Serum")
    raw_text: str = Field(..., description="Customer tweet or product review")

class BatchReviewInput(BaseModel):
    reviews: List[ReviewInput]

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Social Sentiment & Brand Intelligence Engine",
        "architecture": "RoBERTa Aspect-Based Sentiment Analysis (ABSA)",
        "version": "1.0.0"
    }

@app.post("/analyze-single", response_model=ReviewAnalysisResult)
def analyze_single_review(payload: ReviewInput):
    """Analyzes a single review and extracts multi-dimensional aspect sentiments."""
    try:
        return classifier.analyze_review(
            review_id=payload.review_id,
            product_name=payload.product_name,
            raw_text=payload.raw_text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-batch-and-detect-spikes")
def analyze_batch_and_detect_spikes(payload: BatchReviewInput):
    """Processes bulk reviews and automatically generates critical product hazard alerts."""
    try:
        analyzed_items = [
            classifier.analyze_review(
                review_id=r.review_id,
                product_name=r.product_name,
                raw_text=r.raw_text
            )
            for r in payload.reviews
        ]
        alerts = detector.evaluate_batch_for_spikes(analyzed_items)
        return {
            "total_reviews_analyzed": len(analyzed_items),
            "critical_alerts_triggered": len(alerts),
            "alerts": alerts,
            "reviews_breakdown": analyzed_items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8002, reload=True)
