import json
import logging
import math
from pathlib import Path
from typing import List

from policy_simulation.backend.app.core.config import policy_settings
from policy_simulation.backend.app.schemas.policy_case import PolicyCase
from policy_simulation.backend.app.schemas.simulation_result import (
    DimensionScore,
    ImpactAnalysis,
)

logger = logging.getLogger(__name__)

# Approach-based base scores
APPROACH_ECONOMIC_BASE = {
    "mandatory-horizontal": -0.4,
    "mandatory-sectoral": -0.2,
    "principles-based": 0.1,
    "voluntary": 0.3,
}

APPROACH_INNOVATION_BASE = {
    "mandatory-horizontal": -0.2,
    "mandatory-sectoral": -0.1,
    "principles-based": 0.2,
    "voluntary": 0.3,
}

APPROACH_RIGHTS_BASE = {
    "mandatory-horizontal": 0.6,
    "mandatory-sectoral": 0.4,
    "principles-based": 0.2,
    "voluntary": 0.0,
}

# Weights for overall score
DIMENSION_WEIGHTS = {
    "economic": 0.25,
    "innovation": 0.20,
    "rights_protection": 0.25,
    "institutional_capacity": 0.15,
    "international_alignment": 0.15,
}

# Demand levels by approach
APPROACH_DEMAND = {
    "mandatory-horizontal": 0.8,
    "mandatory-sectoral": 0.6,
    "principles-based": 0.3,
    "voluntary": 0.2,
}


class ImpactAnalyzer:
    def __init__(self):
        self.morocco_profile = self._load_morocco_profile()
        self.indicators = self._load_indicators()
        self.morocco_readiness = self.indicators["MA"]["govt_ai_readiness_score"] / 100.0
        self.priority_sectors = set(
            self.morocco_profile["national_ai_strategy"]["priority_sectors"]
        )

    def _load_morocco_profile(self) -> dict:
        profile_path = Path(policy_settings.MOROCCO_PROFILE_FILE)
        with open(profile_path, "r") as f:
            return json.load(f)

    def _load_indicators(self) -> dict:
        indicators_path = Path(policy_settings.INDICATORS_FILE)
        with open(indicators_path, "r") as f:
            return json.load(f)

    def analyze(self, policy: PolicyCase) -> ImpactAnalysis:
        """Rule-based multi-dimensional impact scoring."""
        dimensions = [
            self._score_economic(policy),
            self._score_innovation(policy),
            self._score_rights_protection(policy),
            self._score_institutional_capacity(policy),
            self._score_international_alignment(policy),
        ]

        overall = sum(
            d.score * DIMENSION_WEIGHTS[d.dimension] for d in dimensions
        )

        return ImpactAnalysis(
            dimensions=dimensions,
            overall_score=round(max(-1.0, min(1.0, overall)), 4),
            methodology=(
                "Rule-based impact scoring across 5 dimensions. "
                "Weights: economic 25%, innovation 20%, rights 25%, "
                "institutional 15%, international 15%. "
                "Scores range from -1.0 (negative) to +1.0 (positive)."
            ),
        )

    def _score_economic(self, policy: PolicyCase) -> DimensionScore:
        base = APPROACH_ECONOMIC_BASE.get(policy.approach, 0.0)

        # Cost burden relative to GDP per capita
        cost = policy.costs.annual_compliance_cost_per_system
        if cost > 0:
            source_gdp = policy.costs.source_gdp_per_capita_ppp
            cost_burden = cost / source_gdp if source_gdp > 0 else 0
            # Higher burden → more negative adjustment (capped at -0.3)
            burden_adjustment = -min(cost_burden * 5, 0.3)
            base += burden_adjustment

        explanation = (
            f"{policy.approach} approach has base economic score {APPROACH_ECONOMIC_BASE.get(policy.approach, 0.0):.1f}. "
        )
        if cost > 0:
            explanation += f"Compliance costs ({policy.costs.currency} {cost:,.0f}/system) add economic burden. "
        else:
            explanation += "No mandatory compliance costs. "

        # Morocco SME context
        sme_share = self.morocco_profile["economic_context"]["sme_share_of_economy_percent"]
        if policy.approach in ("mandatory-horizontal", "mandatory-sectoral") and sme_share > 80:
            base -= 0.1
            explanation += f"Morocco's {sme_share}% SME economy increases compliance burden. "

        score = max(-1.0, min(1.0, base))
        return DimensionScore(dimension="economic", score=round(score, 4), explanation=explanation.strip())

    def _score_innovation(self, policy: PolicyCase) -> DimensionScore:
        base = APPROACH_INNOVATION_BASE.get(policy.approach, 0.0)

        # Sandbox bonus
        if policy.provisions.sandbox:
            base += 0.15

        # Maroc IA 2030 sector alignment
        policy_sectors = set(policy.sectors_affected)
        overlap = policy_sectors & self.priority_sectors
        if overlap:
            alignment_bonus = min(len(overlap) * 0.05, 0.2)
            base += alignment_bonus

        explanation = (
            f"{policy.approach} approach has base innovation score {APPROACH_INNOVATION_BASE.get(policy.approach, 0.0):.1f}. "
        )
        if policy.provisions.sandbox:
            explanation += "Sandbox provisions boost innovation (+0.15). "
        if overlap:
            explanation += f"Aligns with Maroc IA 2030 priority sectors: {', '.join(sorted(overlap))}. "

        score = max(-1.0, min(1.0, base))
        return DimensionScore(dimension="innovation", score=round(score, 4), explanation=explanation.strip())

    def _score_rights_protection(self, policy: PolicyCase) -> DimensionScore:
        base = APPROACH_RIGHTS_BASE.get(policy.approach, 0.0)

        # Rights provisions bonus
        if policy.provisions.rights_provisions:
            base += 0.15

        # Penalty deterrence
        if policy.costs.penalty_max_percent_turnover > 0:
            base += 0.1

        # Explainability requirement
        if policy.provisions.explainability_requirement:
            base += 0.1

        explanation = (
            f"{policy.approach} approach has base rights score {APPROACH_RIGHTS_BASE.get(policy.approach, 0.0):.1f}. "
        )
        if policy.provisions.rights_provisions:
            explanation += "Rights provisions strengthen protection (+0.15). "
        if policy.costs.penalty_max_percent_turnover > 0:
            explanation += f"Penalty deterrence ({policy.costs.penalty_max_percent_turnover}% turnover) adds enforcement (+0.1). "
        if policy.provisions.explainability_requirement:
            explanation += "Explainability requirements support transparency (+0.1). "

        score = max(-1.0, min(1.0, base))
        return DimensionScore(dimension="rights_protection", score=round(score, 4), explanation=explanation.strip())

    def _score_institutional_capacity(self, policy: PolicyCase) -> DimensionScore:
        demand = APPROACH_DEMAND.get(policy.approach, 0.5)
        # Gap = what's needed minus what Morocco has
        gap = demand - self.morocco_readiness

        # Positive gap means Morocco lacks capacity → negative score
        # Negative gap means Morocco has capacity → positive score
        score = -gap

        # Audit requirement increases institutional demand
        if policy.provisions.audit_requirement:
            score -= 0.1

        explanation = (
            f"Morocco AI readiness: {self.morocco_readiness:.2f}. "
            f"{policy.approach} approach demands capacity level ~{demand:.1f}. "
        )
        if gap > 0:
            explanation += f"Capacity gap of {gap:.2f} may hinder implementation. "
        else:
            explanation += "Morocco has sufficient baseline capacity for this approach. "
        if policy.provisions.audit_requirement:
            explanation += "Audit requirements add institutional demand. "

        score = max(-1.0, min(1.0, round(score, 4)))
        return DimensionScore(dimension="institutional_capacity", score=score, explanation=explanation.strip())

    def _score_international_alignment(self, policy: PolicyCase) -> DimensionScore:
        score = 0.0
        explanation_parts = []

        # EU alignment bonus (Morocco-EU Association Agreement makes EU alignment valuable)
        if policy.jurisdiction_iso == "EU":
            score += 0.4
            explanation_parts.append(
                "Strong EU alignment bonus (+0.4) due to EU-Morocco Association Agreement and trade relationship."
            )
        elif policy.approach == "mandatory-horizontal":
            # Other mandatory-horizontal frameworks converge with EU norms
            score += 0.2
            explanation_parts.append(
                f"{policy.jurisdiction}'s mandatory-horizontal approach shows norm convergence (+0.2)."
            )
        elif policy.approach in ("mandatory-sectoral", "principles-based"):
            score += 0.1
            explanation_parts.append(
                f"{policy.jurisdiction}'s {policy.approach} approach shows partial norm convergence (+0.1)."
            )

        # UNESCO AI Ethics alignment (Morocco is signatory)
        if policy.provisions.rights_provisions:
            score += 0.1
            explanation_parts.append(
                "Rights provisions align with UNESCO AI Ethics Recommendation (+0.1)."
            )

        if not explanation_parts:
            explanation_parts.append(
                f"{policy.jurisdiction}'s {policy.approach} approach has limited international alignment impact."
            )

        score = max(-1.0, min(1.0, score))
        return DimensionScore(
            dimension="international_alignment",
            score=round(score, 4),
            explanation=" ".join(explanation_parts),
        )
