from typing import List, Dict, Any
from pydantic import BaseModel, Field
from src.aspect_classifier import ReviewAnalysisResult

class AnomalyAlert(BaseModel):
    product_name: str
    target_aspect: str
    complaint_count: int
    severity: str
    recommended_action: str

class TrendAnomalyDetector:
    """
    Monitors velocity of negative sentiment per aspect.
    Triggers automated emergency operational alerts when critical safety/quality spikes occur.
    """
    def __init__(self, spike_threshold: int = 2):
        self.spike_threshold = spike_threshold

    def evaluate_batch_for_spikes(self, analysis_results: List[ReviewAnalysisResult]) -> List[AnomalyAlert]:
        alerts = []
        product_aspect_counter: Dict[str, Dict[str, int]] = {}

        for item in analysis_results:
            p_name = item.product_name
            if p_name not in product_aspect_counter:
                product_aspect_counter[p_name] = {}

            for aspect_item in item.aspects_detected:
                if aspect_item.sentiment == "NEGATIVE":
                    asp = aspect_item.aspect
                    product_aspect_counter[p_name][asp] = product_aspect_counter[p_name].get(asp, 0) + 1

        # Check thresholds
        for p_name, aspects in product_aspect_counter.items():
            for asp, count in aspects.items():
                if count >= self.spike_threshold:
                    if asp == "SKIN_REACTION":
                        severity = "CRITICAL_HAZARD"
                        action = "TRIGGER_IMMEDIATE_BATCH_QUARANTINE & DISPATCH_QUALITY_LAB_AUDIT"
                    elif asp == "PACKAGING":
                        severity = "MODERATE_LOGISTICS_DEFECT"
                        action = "NOTIFY_FULFILLMENT_CENTER_TO_INSPECT_BUBBLE_WRAP_PROTOCOL"
                    else:
                        severity = "ELEVATED_DISSATISFACTION"
                        action = "ESCALATE_TO_BRAND_RELATIONS_TEAM"

                    alerts.append(AnomalyAlert(
                        product_name=p_name,
                        target_aspect=asp,
                        complaint_count=count,
                        severity=severity,
                        recommended_action=action
                    ))

        return alerts
