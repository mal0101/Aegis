import logging

from fastapi import APIRouter, HTTPException

from policy_simulation.backend.app.schemas.simulation_request import SimulationRequest
from policy_simulation.backend.app.schemas.simulation_result import SimulationResult
from policy_simulation.backend.app.services.simulation_engine import SimulationEngine

logger = logging.getLogger(__name__)

router = APIRouter()

_engine: SimulationEngine | None = None


def get_engine() -> SimulationEngine:
    global _engine
    if _engine is None:
        _engine = SimulationEngine()
    return _engine


@router.post("/simulate", response_model=SimulationResult)
async def simulate_policy(request: SimulationRequest):
    """Simulate the impact of an AI policy proposal for Morocco."""
    try:
        engine = get_engine()
        result = engine.simulate(
            policy_description=request.policy_description,
            target_sectors=request.target_sectors,
            num_matches=request.num_matches,
        )
        return result
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
