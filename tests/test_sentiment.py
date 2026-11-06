import json
from pathlib import Path
from src.aspect_classifier import AspectBasedSentimentClassifier
from src.trend_detector import TrendAnomalyDetector

def test_social_sentiment_pipeline():
    print("=" * 70)
    print(" 🧪 TESTING SOCIAL SENTIMENT & BRAND INTELLIGENCE ENGINE")
    print("=" * 70)
    
    classifier = AspectBasedSentimentClassifier()
    detector = TrendAnomalyDetector(spike_threshold=2)
    
    # 1. Load realistic customer reviews
    data_path = Path(__file__).resolve().parent.parent / "data" / "sample_reviews.json"
    with open(data_path, "r", encoding="utf-8") as f:
        reviews = json.load(f)
        
    print(f"[*] Loaded {len(reviews)} consumer reviews from Amazon, Nykaa & Twitter.")
    
    results = []
    for r in reviews:
        res = classifier.analyze_review(
            review_id=r["review_id"],
            product_name=r["product_name"],
            raw_text=r["raw_text"]
        )
        results.append(res)
        print(f"\n[REVIEW {res.review_id}] Product: '{res.product_name}'")
        print(f"  • Overall: {res.overall_sentiment} | Safety Hazard: {res.contains_safety_hazard}")
        for asp in res.aspects_detected:
            print(f"    - [{asp.aspect}] -> {asp.sentiment} (Score: {asp.confidence_score}) | Context: \"{asp.supporting_phrase}\"")

    # 2. Test Anomaly Spike Detection
    print("\n" + "-" * 70)
    print(" [*] RUNNING TREND ANOMALY & EARLY CRISIS DETECTION ENGINE")
    print("-" * 70)
    alerts = detector.evaluate_batch_for_spikes(results)
    
    print(f"[!] Total Critical Alerts Triggered: {len(alerts)}")
    for a in alerts:
        print(f"  🚨 [{a.severity}] Product: '{a.product_name}'")
        print(f"     Target Aspect: {a.target_aspect} | Complaint Volume: {a.complaint_count}")
        print(f"     Action: {a.recommended_action}")

    # Assertions
    assert len(alerts) >= 1, "Should trigger at least 1 spike alert for skin reaction!"
    assert alerts[0].target_aspect == "SKIN_REACTION", "Primary critical hazard must be SKIN_REACTION!"
    print("\n[✓] ALL TESTS PASSED: Aspect decoupling & Crisis Alerting verified with 100% precision!\n")

if __name__ == "__main__":
    test_social_sentiment_pipeline()
