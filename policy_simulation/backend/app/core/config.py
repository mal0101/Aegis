from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class PolicySimulationSettings(BaseSettings):
    POLICY_CHROMADB_PATH: str = Field(
        default="./storage/policy_chromadb",
        description="File path for storing policy ChromaDB data.",
    )
    POLICY_CHROMADB_COLLECTION: str = Field(
        default="policy_cases",
        description="ChromaDB collection name for policy cases.",
    )
    POLICIES_DIR: str = Field(
        default=str(
            Path(__file__).parent.parent / "data" / "policies"
        ),
        description="Directory containing policy JSON files.",
    )
    INDICATORS_FILE: str = Field(
        default=str(
            Path(__file__).parent.parent / "data" / "indicators" / "economic_indicators.json"
        ),
        description="Path to economic indicators JSON file.",
    )
    MOROCCO_PROFILE_FILE: str = Field(
        default=str(
            Path(__file__).parent.parent / "data" / "morocco_profile.json"
        ),
        description="Path to Morocco profile JSON file.",
    )
    USD_TO_MAD: float = Field(
        default=10.0,
        description="USD to MAD exchange rate.",
    )

    model_config = {
        "env_prefix": "POLICY_SIM_",
        "case_sensitive": True,
    }


policy_settings = PolicySimulationSettings()
