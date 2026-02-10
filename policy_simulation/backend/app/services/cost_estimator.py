import json
import logging
from pathlib import Path
from typing import Optional

from policy_simulation.backend.app.core.config import policy_settings
from policy_simulation.backend.app.schemas.policy_case import PolicyCase
from policy_simulation.backend.app.schemas.simulation_result import CostEstimate

logger = logging.getLogger(__name__)


class CostEstimator:
    def __init__(self):
        self.indicators = self._load_indicators()
        self.morocco_gdp_ppp = self.indicators["MA"]["gdp_per_capita_ppp"]
        self.exchange_rates = self.indicators["exchange_rates_to_usd"]

    def _load_indicators(self) -> dict:
        indicators_path = Path(policy_settings.INDICATORS_FILE)
        with open(indicators_path, "r") as f:
            return json.load(f)

    def estimate(self, policy: PolicyCase) -> Optional[CostEstimate]:
        """Estimate Morocco-adjusted costs from a reference policy using PPP adjustment."""
        source_cost = policy.costs.annual_compliance_cost_per_system
        sme_cost = policy.costs.sme_cost_per_system
        source_gdp_ppp = policy.costs.source_gdp_per_capita_ppp

        # If no compliance costs (voluntary/in-progress), return minimal estimate
        if source_cost == 0 and sme_cost == 0:
            return CostEstimate(
                annual_cost_per_system_mad=0,
                annual_cost_per_system_usd=0,
                sme_cost_per_system_mad=0,
                sme_cost_per_system_usd=0,
                ppp_adjustment_factor=0,
                source_policy=policy.name,
                methodology=f"No mandatory compliance costs. {policy.name} is a {policy.approach} framework.",
            )

        # Convert source cost to USD
        currency = policy.costs.currency
        exchange_rate = self.exchange_rates.get(currency, 1.0)
        source_cost_usd = source_cost * exchange_rate
        sme_cost_usd = sme_cost * exchange_rate

        # PPP adjustment
        ppp_factor = self.morocco_gdp_ppp / source_gdp_ppp

        # Adjusted costs in USD
        morocco_cost_usd = source_cost_usd * ppp_factor
        morocco_sme_cost_usd = sme_cost_usd * ppp_factor

        # Convert to MAD
        usd_to_mad = policy_settings.USD_TO_MAD
        morocco_cost_mad = morocco_cost_usd * usd_to_mad
        morocco_sme_cost_mad = morocco_sme_cost_usd * usd_to_mad

        return CostEstimate(
            annual_cost_per_system_mad=round(morocco_cost_mad, 2),
            annual_cost_per_system_usd=round(morocco_cost_usd, 2),
            sme_cost_per_system_mad=round(morocco_sme_cost_mad, 2),
            sme_cost_per_system_usd=round(morocco_sme_cost_usd, 2),
            ppp_adjustment_factor=round(ppp_factor, 4),
            source_policy=policy.name,
            methodology=(
                f"PPP-adjusted from {policy.name} ({policy.jurisdiction}). "
                f"Source cost: {currency} {source_cost:,.0f}/system/year. "
                f"PPP factor (Morocco/Source): {ppp_factor:.4f}. "
                f"Source: {policy.costs.cost_source}."
            ),
        )
