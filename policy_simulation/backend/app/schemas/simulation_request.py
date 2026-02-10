from pydantic import BaseModel, Field
from typing import Optional, List


class SimulationRequest(BaseModel):
    policy_description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Description of the AI policy to simulate.",
    )
    target_sectors: Optional[List[str]] = Field(
        default=None,
        description="Specific sectors to focus the simulation on. If None, inferred from matched policies.",
    )
    num_matches: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of reference policies to match against.",
    )
