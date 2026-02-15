import logging
from openai import OpenAI
from app.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self._model = LLM_MODEL
        if not LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY not set. Add it to your .env file.")
        self._client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY
        )
        self._verify_connection()
        logger.info(f"LLM service initialized: {self._model} via {LLM_BASE_URL}")

    def _verify_connection(self):
        """Verify the API is reachable with a lightweight models.list call."""
        try:
            self._client.models.list()
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to LLM API at {LLM_BASE_URL}: {e}"
            ) from e

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature or LLM_TEMPERATURE,
                max_tokens=max_tokens or LLM_MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise


# Singleton
llm_service = LLMService()
