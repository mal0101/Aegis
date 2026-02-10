import json
import logging
import math
from pathlib import Path
from typing import Optional

from policy_simulation.backend.app.core.config import policy_settings
from policy_simulation.backend.app.schemas.policy_case import PolicyCase
from policy_simulation.backend.app.schemas.simulation_result import TimelineEstimate

logger = logging.getLogger(__name__)


class TimelineEstimator:
    def __init__(self):
        self.indicators = self._load_indicators()
        self.morocco_readiness = self.indicators["MA"]["govt_ai_readiness_score"]

    def _load_indicators(self) -> dict:
        indicators_path = Path(policy_settings.INDICATORS_FILE)
        with open(indicators_path, "r") as f:
            return json.load(f)

    def estimate(self, policy: PolicyCase) -> Optional[TimelineEstimate]:
        """Estimate Morocco-adjusted timeline from a reference policy using readiness adjustment."""
        source_iso = policy.jurisdiction_iso
        source_indicators = self.indicators.get(source_iso)

        if source_indicators is None:
            logger.warning(f"No indicators found for {source_iso}")
            return None

        source_readiness = source_indicators["govt_ai_readiness_score"]
        readiness_ratio = source_readiness / self.morocco_readiness
        # Clamp to prevent absurd values
        readiness_ratio = max(0.5, min(readiness_ratio, 2.5))

        # Adjust announcement-to-enactment timeline
        source_enactment_years = policy.timeline.announcement_to_enactment_years
        morocco_enactment_years = None
        if source_enactment_years is not None:
            morocco_enactment_years = round(source_enactment_years * readiness_ratio, 1)

        # Adjust enactment-to-enforcement timeline (softer adjustment using sqrt)
        source_enforcement_years = policy.timeline.enactment_to_enforcement_years
        morocco_enforcement_years = None
        if source_enforcement_years is not None:
            morocco_enforcement_years = round(
                source_enforcement_years * math.sqrt(readiness_ratio), 1
            )

        # Total
        morocco_total = None
        if morocco_enactment_years is not None and morocco_enforcement_years is not None:
            morocco_total = round(morocco_enactment_years + morocco_enforcement_years, 1)
        elif morocco_enactment_years is not None:
            morocco_total = morocco_enactment_years

        return TimelineEstimate(
            announcement_to_enactment_years=morocco_enactment_years,
            enactment_to_enforcement_years=morocco_enforcement_years,
            total_years=morocco_total,
            readiness_adjustment_factor=round(readiness_ratio, 4),
            source_policy=policy.name,
            methodology=(
                f"Readiness-adjusted from {policy.name} ({policy.jurisdiction}). "
                f"Source readiness: {source_readiness}, Morocco readiness: {self.morocco_readiness}. "
                f"Ratio: {readiness_ratio:.4f}x. "
                f"Enactment phase uses linear scaling, enforcement phase uses sqrt scaling."
            ),
        )
