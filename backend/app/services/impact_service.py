import json
import time
import logging
from typing import List, Dict, Any

from app.config import MOROCCO_CONTEXT_FILE

logger = logging.getLogger(__name__)

POLICY_TEMPLATES = [
    {
        "id": "bias_testing",
        "name": "Mandatory Bias Testing",
        "description": "Require all AI systems used in public services to undergo bias testing before deployment, with annual re-evaluation.",
        "sectors": ["public_services", "healthcare", "education"],
        "policy_type": "sectoral"
    },
    {
        "id": "transparency_requirement",
        "name": "AI Transparency Requirement",
        "description": "Mandatory disclosure when AI is used in decisions affecting citizens. Public sector AI systems must provide explanations for automated decisions.",
        "sectors": ["public_services", "finance"],
        "policy_type": "sectoral"
    },
    {
        "id": "ai_sandbox",
        "name": "AI Regulatory Sandbox",
        "description": "Create a controlled environment where AI startups can test innovations with relaxed regulations for 2 years, with CNDP oversight.",
        "sectors": ["finance", "healthcare", "agriculture"],
        "policy_type": "sandbox"
    },
    {
        "id": "data_governance",
        "name": "National Data Governance Framework",
        "description": "Establish data sharing standards, quality requirements, and governance structures for public sector data used to train AI systems.",
        "sectors": ["public_services", "healthcare", "education", "agriculture"],
        "policy_type": "comprehensive"
    },
    {
        "id": "comprehensive_ai_act",
        "name": "Morocco AI Act",
        "description": "Comprehensive risk-based AI regulation modeled on the EU AI Act but adapted for Morocco's economic context and Digital Morocco 2030 strategy.",
        "sectors": ["all"],
        "policy_type": "comprehensive"
    }
]


class ImpactService:
    def __init__(self):
        self._morocco_context = self._load_morocco_context()
        logger.info("Impact service initialized")

    def _load_morocco_context(self) -> dict:
        try:
            with open(MOROCCO_CONTEXT_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load Morocco context: {e}")
            return {}

    def get_templates(self) -> List[Dict]:
        return POLICY_TEMPLATES

    def predict(self, policy_name: str, description: str, sectors: list, policy_type: str) -> dict:
        start = time.time()

        from app.services.case_study_service import case_study_service
        from app.services.llm_service import llm_service

        # Find similar international policies
        similar = case_study_service.find_similar(
            description=description,
            policy_type=policy_type,
            top_k=5
        )

        # Aggregate outcome metrics from similar policies
        impact_dimensions = self._calculate_impact_dimensions(similar, sectors, policy_type)

        # Build evidence summary for LLM
        evidence_summary = self._build_evidence_summary(similar)
        recommendations = self._generate_recommendations(impact_dimensions, sectors)

        # Generate narrative analysis via LLM
        system_prompt = (
            "You are an AI policy impact analyst for Morocco. "
            "Analyze the predicted impacts of a proposed policy based on international evidence. "
            "Be specific about Morocco's context: GDP $3,600/capita, Law 09-08 data protection, "
            "Digital Morocco 2030 strategy, French civil law tradition. "
            "Structure your response with an executive summary paragraph followed by detailed analysis."
        )

        morocco_gdp = self._morocco_context.get("economy", {}).get("gdp_per_capita_usd", 3600)
        user_prompt = (
            f"Proposed policy: {policy_name}\n"
            f"Description: {description}\n"
            f"Target sectors: {', '.join(sectors)}\n"
            f"Policy type: {policy_type}\n\n"
            f"International evidence from {len(similar)} similar policies:\n{evidence_summary}\n\n"
            f"Morocco GDP/capita: ${morocco_gdp}\n\n"
            f"Predicted impact scores:\n"
            + "\n".join(f"- {d['name']}: {d['score']:.1f}/10 ({d['confidence']} confidence)" for d in impact_dimensions)
            + "\n\nProvide:\n1. Executive summary (2-3 sentences)\n2. Detailed analysis of each impact dimension\n3. Key risks and opportunities for Morocco"
        )

        try:
            analysis = llm_service.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=2048
            )
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            analysis = "LLM analysis unavailable. See quantitative impact dimensions below."

        # Parse executive summary from analysis
        lines = analysis.split("\n")
        executive_summary = ""
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("*"):
                executive_summary = line
                break
        if not executive_summary:
            executive_summary = f"Analysis of {policy_name} based on {len(similar)} similar international policies."

        # Build similar policy summaries
        similar_summaries = [
            {
                "id": s["id"],
                "country": s["country"],
                "policy_name": s["policy"]["name"],
                "policy_type": s["policy"].get("type", ""),
                "enacted_date": s["policy"].get("enacted_date", ""),
                "data_quality": s["outcomes"].get("data_quality", "medium"),
                "tags": s["metadata"].get("tags", []),
                "relevance_score": s.get("relevance_score", 0.0)
            }
            for s in similar
        ]

        elapsed = int((time.time() - start) * 1000)

        return {
            "policy_name": policy_name,
            "executive_summary": executive_summary,
            "full_analysis": analysis,
            "similar_policies": similar_summaries,
            "impact_dimensions": impact_dimensions,
            "recommendations": recommendations,
            "evidence_base_size": len(similar),
            "processing_time_ms": elapsed
        }

    def _calculate_impact_dimensions(self, similar: list, sectors: list, policy_type: str) -> List[Dict]:
        """Calculate 5 impact dimensions from aggregated evidence."""
        morocco_gdp = self._morocco_context.get("economy", {}).get("gdp_per_capita_usd", 3600)

        # Aggregate metrics
        trust_scores = []
        bias_scores = []
        cost_scores = []
        startup_scores = []
        timeline_scores = []
        compliance_scores = []

        for s in similar:
            outcomes = s.get("outcomes", {})
            social = outcomes.get("social_impact", {})
            economic = outcomes.get("economic_impact", {})
            implementation = outcomes.get("implementation_reality", {})
            meta = s.get("metadata", {})

            gdp_ratio = meta.get("gdp_ratio_to_morocco", 1.0)
            legal_sim = meta.get("legal_similarity", 0.5)

            # Weight by legal similarity
            weight = legal_sim

            if social.get("trust_change_pct") is not None:
                trust_scores.append((social["trust_change_pct"] * weight, weight))
            if social.get("bias_reduction_pct") is not None:
                bias_scores.append((social["bias_reduction_pct"] * weight, weight))
            if economic.get("compliance_costs_usd") is not None:
                # Adjust cost by GDP ratio (PPP approximation)
                adjusted_cost = economic["compliance_costs_usd"] / max(gdp_ratio, 0.1)
                cost_scores.append((adjusted_cost, weight))
            if economic.get("startup_growth_pct") is not None:
                startup_scores.append((economic["startup_growth_pct"] * weight, weight))
            if implementation.get("timeline_months") is not None:
                timeline_scores.append((implementation["timeline_months"], weight))
            if implementation.get("compliance_rate_pct") is not None:
                compliance_scores.append((implementation["compliance_rate_pct"] * weight, weight))

        def weighted_avg(scores):
            if not scores:
                return 0.0
            total_val = sum(v for v, _ in scores)
            total_w = sum(w for _, w in scores)
            return total_val / total_w if total_w > 0 else 0.0

        # Calculate dimension scores (normalized to 0-10)
        avg_trust = weighted_avg(trust_scores)
        avg_bias = weighted_avg(bias_scores)
        avg_startup = weighted_avg(startup_scores)
        avg_cost = sum(c for c, _ in cost_scores) / len(cost_scores) if cost_scores else 0
        avg_timeline = sum(t for t, _ in timeline_scores) / len(timeline_scores) if timeline_scores else 18
        avg_compliance = weighted_avg(compliance_scores)

        countries_evidence = list(set(s["country"] for s in similar))

        dimensions = [
            {
                "name": "Social Trust",
                "score": min(10, max(0, avg_trust / 10)),
                "confidence": "high" if len(trust_scores) >= 3 else "medium" if trust_scores else "low",
                "explanation": f"Expected {avg_trust:.0f}% improvement in public trust based on {len(trust_scores)} comparable policies.",
                "evidence_countries": countries_evidence
            },
            {
                "name": "Bias Reduction",
                "score": min(10, max(0, avg_bias / 10)),
                "confidence": "high" if len(bias_scores) >= 3 else "medium" if bias_scores else "low",
                "explanation": f"Expected {avg_bias:.0f}% reduction in documented AI bias incidents.",
                "evidence_countries": countries_evidence
            },
            {
                "name": "Economic Impact",
                "score": min(10, max(0, 5 + avg_startup / 10)),
                "confidence": "medium" if startup_scores else "low",
                "explanation": f"Expected {avg_startup:.0f}% change in AI startup growth. Morocco-adjusted compliance cost: ~${avg_cost:,.0f}/company.",
                "evidence_countries": countries_evidence
            },
            {
                "name": "Implementation Feasibility",
                "score": min(10, max(0, avg_compliance / 10)),
                "confidence": "medium" if compliance_scores else "low",
                "explanation": f"Expected {avg_timeline:.0f}-month implementation timeline with {avg_compliance:.0f}% compliance rate.",
                "evidence_countries": countries_evidence
            },
            {
                "name": "Innovation Environment",
                "score": min(10, max(0, 5 + avg_startup / 20)),
                "confidence": "medium" if startup_scores else "low",
                "explanation": f"Net effect on Morocco's AI ecosystem of {self._morocco_context.get('ai_ecosystem', {}).get('ai_startups', 45)} startups.",
                "evidence_countries": countries_evidence
            }
        ]

        return dimensions

    def _build_evidence_summary(self, similar: list) -> str:
        """Build a text summary of evidence for the LLM prompt."""
        parts = []
        for s in similar:
            outcomes = s.get("outcomes", {})
            social = outcomes.get("social_impact", {})
            economic = outcomes.get("economic_impact", {})
            impl = outcomes.get("implementation_reality", {})

            parts.append(
                f"- {s['policy']['name']} ({s['country']}): "
                f"trust +{social.get('trust_change_pct', 'N/A')}%, "
                f"bias -{social.get('bias_reduction_pct', 'N/A')}%, "
                f"compliance cost ${economic.get('compliance_costs_usd', 'N/A')}, "
                f"startup growth {economic.get('startup_growth_pct', 'N/A')}%, "
                f"timeline {impl.get('timeline_months', 'N/A')}mo, "
                f"compliance {impl.get('compliance_rate_pct', 'N/A')}%"
            )
        return "\n".join(parts) if parts else "No comparable evidence found."

    def _generate_recommendations(self, dimensions: list, sectors: list) -> List[str]:
        """Generate rule-based recommendations from impact scores."""
        recs = []

        for dim in dimensions:
            if dim["name"] == "Implementation Feasibility" and dim["score"] < 5:
                recs.append("Consider a phased rollout starting with a pilot in one sector before expanding nationwide.")
            if dim["name"] == "Economic Impact" and dim["score"] < 4:
                recs.append("Include compliance cost subsidies for SMEs and startups to prevent market exit.")
            if dim["name"] == "Social Trust" and dim["score"] > 5:
                recs.append("Leverage the expected trust improvement for public communication campaigns.")
            if dim["name"] == "Innovation Environment" and dim["score"] < 4:
                recs.append("Pair regulation with an AI regulatory sandbox to protect innovation.")

        if "agriculture" in sectors:
            recs.append("Engage JAZARI Institute and agricultural cooperatives for sector-specific AI standards.")
        if "healthcare" in sectors:
            recs.append("Coordinate with Ministry of Health for AI medical device classification aligned with WHO guidelines.")
        if "finance" in sectors:
            recs.append("Align with Bank Al-Maghrib digital finance regulations and CNDP requirements.")

        if not recs:
            recs.append("Conduct stakeholder consultations with Morocco's AI ecosystem before finalizing implementation details.")

        return recs[:6]


# Singleton
impact_service = ImpactService()
