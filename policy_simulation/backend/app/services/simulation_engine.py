import json
import logging
from pathlib import Path
from typing import List, Optional

from concept_translator.backend.app.services.llm import LLMService
from policy_simulation.backend.app.core.config import policy_settings
from policy_simulation.backend.app.schemas.policy_case import PolicyCase
from policy_simulation.backend.app.schemas.simulation_result import (
    CostEstimate,
    ImpactAnalysis,
    MatchedPolicy,
    SimulationResult,
    TimelineEstimate,
)
from policy_simulation.backend.app.services.cost_estimator import CostEstimator
from policy_simulation.backend.app.services.impact_analyzer import ImpactAnalyzer
from policy_simulation.backend.app.services.policy_matcher import PolicyMatcher
from policy_simulation.backend.app.services.policy_vector_db import PolicyVectorDBService
from policy_simulation.backend.app.services.timeline_estimator import TimelineEstimator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an AI policy analyst specializing in Morocco's regulatory landscape.
You synthesize simulation results into clear, actionable narratives for Moroccan policymakers.

CRITICAL RULES:
- NEVER generate or invent numbers. All cost figures, timelines, and scores are provided in the data below.
- Your role is ONLY to explain and contextualize the numbers that are given to you.
- Reference specific data points from the simulation results.
- Connect findings to Morocco's context: Maroc IA 2030, CNDP, Law 09-08, EU-Morocco trade relationship.
- Write in clear, accessible language for policymakers without technical backgrounds.
- Structure your narrative with clear sections.
- Keep the narrative concise (300-500 words).
"""


class SimulationEngine:
    def __init__(self):
        self.vector_db = PolicyVectorDBService()
        self.matcher = PolicyMatcher(self.vector_db)
        self.cost_estimator = CostEstimator()
        self.timeline_estimator = TimelineEstimator()
        self.impact_analyzer = ImpactAnalyzer()
        self.llm = LLMService()

    def simulate(
        self,
        policy_description: str,
        target_sectors: Optional[List[str]] = None,
        num_matches: int = 3,
    ) -> SimulationResult:
        """Run full simulation pipeline."""
        # Step 1: Match similar policies
        matched_policies = self.matcher.match(policy_description, n_results=num_matches)

        if not matched_policies:
            return self._empty_result(policy_description)

        # Step 2: Get the top match for detailed analysis
        top_match = matched_policies[0]
        top_policy = self.matcher.get_full_policy(top_match.id)

        # Step 3: Cost estimation (from top match with actual costs)
        cost_estimate = self._get_best_cost_estimate(matched_policies)

        # Step 4: Timeline estimation (from top match with actual timeline)
        timeline_estimate = self._get_best_timeline_estimate(matched_policies)

        # Step 5: Impact analysis (from top match)
        impact_analysis = self.impact_analyzer.analyze(top_policy)

        # Step 6: Generate narrative (LLM synthesizes, never generates numbers)
        narrative = self._generate_narrative(
            policy_description,
            matched_policies,
            cost_estimate,
            timeline_estimate,
            impact_analysis,
        )

        # Collect data sources
        data_sources = [
            "World Bank WDI 2024 (GDP per capita PPP)",
            "Oxford Insights Government AI Readiness Index 2025",
            "World Bank Worldwide Governance Indicators 2023",
        ]
        for mp in matched_policies:
            full = self.matcher.get_full_policy(mp.id)
            data_sources.append(f"{full.name}: {full.costs.cost_source}")

        return SimulationResult(
            policy_description=policy_description,
            matched_policies=matched_policies,
            cost_estimate=cost_estimate,
            timeline_estimate=timeline_estimate,
            impact_analysis=impact_analysis,
            narrative=narrative,
            data_sources=list(dict.fromkeys(data_sources)),  # deduplicate preserving order
        )

    def _get_best_cost_estimate(self, matches: List[MatchedPolicy]) -> Optional[CostEstimate]:
        """Find the best cost estimate from matched policies (prefer ones with actual costs)."""
        for match in matches:
            policy = self.matcher.get_full_policy(match.id)
            if policy.costs.annual_compliance_cost_per_system > 0:
                return self.cost_estimator.estimate(policy)
        # Fall back to top match even if no costs
        top_policy = self.matcher.get_full_policy(matches[0].id)
        return self.cost_estimator.estimate(top_policy)

    def _get_best_timeline_estimate(self, matches: List[MatchedPolicy]) -> Optional[TimelineEstimate]:
        """Find the best timeline estimate from matched policies (prefer enacted ones)."""
        for match in matches:
            policy = self.matcher.get_full_policy(match.id)
            if policy.timeline.announcement_to_enactment_years is not None:
                return self.timeline_estimator.estimate(policy)
        # Fall back to top match
        top_policy = self.matcher.get_full_policy(matches[0].id)
        return self.timeline_estimator.estimate(top_policy)

    def _generate_narrative(
        self,
        policy_description: str,
        matched_policies: List[MatchedPolicy],
        cost_estimate: Optional[CostEstimate],
        timeline_estimate: Optional[TimelineEstimate],
        impact_analysis: ImpactAnalysis,
    ) -> str:
        """Use LLM to synthesize a narrative from simulation data. LLM never generates numbers."""
        user_prompt = self._build_user_prompt(
            policy_description, matched_policies, cost_estimate, timeline_estimate, impact_analysis
        )
        result = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        return result

    def _build_user_prompt(
        self,
        policy_description: str,
        matched_policies: List[MatchedPolicy],
        cost_estimate: Optional[CostEstimate],
        timeline_estimate: Optional[TimelineEstimate],
        impact_analysis: ImpactAnalysis,
    ) -> str:
        sections = [f"## Policy Proposal\n{policy_description}\n"]

        # Matched policies
        matches_text = "\n".join(
            f"- {m.name} ({m.jurisdiction}, {m.approach}) — similarity: {m.similarity_score:.2f}"
            for m in matched_policies
        )
        sections.append(f"## Similar Reference Policies\n{matches_text}\n")

        # Costs
        if cost_estimate:
            sections.append(
                f"## Cost Estimate for Morocco\n"
                f"- Annual cost per system: {cost_estimate.annual_cost_per_system_mad:,.0f} MAD "
                f"({cost_estimate.annual_cost_per_system_usd:,.0f} USD)\n"
                f"- SME cost per system: {cost_estimate.sme_cost_per_system_mad:,.0f} MAD "
                f"({cost_estimate.sme_cost_per_system_usd:,.0f} USD)\n"
                f"- PPP adjustment factor: {cost_estimate.ppp_adjustment_factor:.4f}\n"
                f"- Methodology: {cost_estimate.methodology}\n"
            )

        # Timeline
        if timeline_estimate:
            parts = [f"## Timeline Estimate for Morocco\n"]
            if timeline_estimate.announcement_to_enactment_years is not None:
                parts.append(
                    f"- Announcement to enactment: {timeline_estimate.announcement_to_enactment_years:.1f} years\n"
                )
            if timeline_estimate.enactment_to_enforcement_years is not None:
                parts.append(
                    f"- Enactment to full enforcement: {timeline_estimate.enactment_to_enforcement_years:.1f} years\n"
                )
            if timeline_estimate.total_years is not None:
                parts.append(f"- Total estimated duration: {timeline_estimate.total_years:.1f} years\n")
            parts.append(
                f"- Readiness adjustment factor: {timeline_estimate.readiness_adjustment_factor:.4f}x\n"
                f"- Methodology: {timeline_estimate.methodology}\n"
            )
            sections.append("".join(parts))

        # Impact
        dims_text = "\n".join(
            f"- {d.dimension}: {d.score:+.4f} — {d.explanation}" for d in impact_analysis.dimensions
        )
        sections.append(
            f"## Impact Analysis\n{dims_text}\n"
            f"- Overall score: {impact_analysis.overall_score:+.4f}\n"
        )

        sections.append(
            "## Task\n"
            "Synthesize the above data into a clear narrative for Moroccan policymakers. "
            "DO NOT invent any numbers — only reference the data provided above. "
            "Explain what the numbers mean, highlight key risks and opportunities, "
            "and connect findings to Morocco's Maroc IA 2030 strategy and regulatory context."
        )

        return "\n".join(sections)

    def _empty_result(self, policy_description: str) -> SimulationResult:
        return SimulationResult(
            policy_description=policy_description,
            matched_policies=[],
            cost_estimate=None,
            timeline_estimate=None,
            impact_analysis=ImpactAnalysis(
                dimensions=[],
                overall_score=0.0,
                methodology="No matching policies found for analysis.",
            ),
            narrative="No similar reference policies were found. Please try a more specific policy description.",
            data_sources=[],
        )
