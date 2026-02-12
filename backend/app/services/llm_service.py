import logging
from openai import OpenAI
from app.config import OLLAMA_BASE_URL, LLM_MODEL, LLM_FALLBACK_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self._model = LLM_MODEL
        self._client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama"  # Ollama doesn't need a real key but the SDK requires the field
        )
        self._verify_connection()
        logger.info(f"LLM service initialized: {self._model} via Ollama")

    def _verify_connection(self):
        """Check that Ollama is running and the model is pulled."""
        try:
            models = self._client.models.list()
            available = [m.id for m in models.data]
            logger.info(f"Ollama models available: {available}")

            if self._model not in available and f"{self._model}:latest" not in available:
                matching = [m for m in available if m.startswith(self._model)]
                if not matching:
                    logger.warning(f"Model '{self._model}' not found. Available: {available}")
                    fallback_match = [m for m in available if m.startswith(LLM_FALLBACK_MODEL)]
                    if fallback_match:
                        self._model = LLM_FALLBACK_MODEL
                        logger.info(f"Using fallback model: {self._model}")
                    else:
                        raise RuntimeError(
                            f"Neither '{LLM_MODEL}' nor '{LLM_FALLBACK_MODEL}' found in Ollama. "
                            f"Run: ollama pull {LLM_MODEL}"
                        )
        except Exception as e:
            if "Connection" in str(type(e).__name__) or "refused" in str(e).lower():
                raise RuntimeError(
                    "Ollama is not running. Start it with: ollama serve\n"
                    "Install Ollama: curl -fsSL https://ollama.com/install.sh | sh\n"
                    f"Then pull the model: ollama pull {LLM_MODEL}"
                ) from e
            raise

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
