from groq import Groq
from typing import Optional, Dict, Any, List
import logging
import time
from concept_translator.backend.app.core.config import settings
class LLMService:
    def __init__(self):
        if settings.LLM_PROVIDER != "groq":
            raise RuntimeError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in environment variables")
        self.model = settings.LLM_MODEL
        try:
            self.client = Groq(api_key=settings.GROQ_API_KEY)
        except Exception as e:
            logging.error(f"Failed to initialize Groq client: {e}")
            raise RuntimeError("Failed to initialize LLM client") from e
        
    
    def generate(self, system_prompt, user_prompt):
        try:
            messages = [
                {
                    "role":"system",
                    "content": system_prompt
                },
                {
                    "role":"user",
                    "content": user_prompt
                }
            ]
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                top_p=1,
                stream=False
            )
            duration = time.time() - start_time
            generated_text = response.choices[0].message.content
            return generated_text
        except Exception as e:
            return self._handle_error(e)
        
    
    def generate_with_context(self, question, context, include_morocco_focus=True):
        system_prompt = self._build_system_prompt(include_morocco_focus)
        user_prompt = f"""
        Based on this information: {context}
        Question: {question}
        Please provide a clear, comprehensive answer that:
        1. Explains the concept accurately
        2. Includes specific and relevant examples
        3. Highlights implications for Morocco's AI development and policy
        4. Is accessible to policymakers without technical backgrounds or necessarily prior knowledge of the concepts at hand
        """
        return self.generate(system_prompt, user_prompt)
    
    def _build_system_prompt(self, include_morocco_focus=True):
        if include_morocco_focus:
            return """You are an expert AI policy advisor specializing in Morocco's artificial intelligence landscape and regulatory needs.
        Your role is to explain AI concepts clearly and actionably for Moroccan policymakers, with specific attention to:
        Morocco's AI Context:
        - Digital Morocco 2030 strategy (national digitalization initiative)
        - JAZARI AI Institutes (government AI development program)
        - UNESCO partnership on AI governance
        - Morocco-EU trade relationships requiring regulatory alignment
        - Regional diversity (urban Casablanca to rural Atlas Mountains)
        - Linguistic diversity (Arabic, French, Amazigh languages, Darija dialect)
        - Socioeconomic disparities requiring inclusive AI development
        
        Your Communication Style:
        - Clear and accessible language (avoid unnecessary jargon)
        - Concrete examples over abstract concepts
        - Specific recommendations, not just general observations
        - Balance technical accuracy with policy relevance
        - Acknowledge when Morocco lacks specific regulations and/or infrastructure
        
        Your Approach:
        When answering questions:
        1. Define the concept clearly
        2. Explain why it matters specifically for Morocco
        3. Provide concrete examples (inclusing Morocco-relevant cases when possible)
        4. Suggest actionable implications for Moroccan policy or instituions
        5. Connect to Morocco's broader AI development goals
        
        Remember: Your audience is policymakers who may not have technical backgrounds but need to make informed decisions about AI governance, investment, and regulation.
        """
        else:
            return """You are an expert AI policy advisor.
        Explain concepts clearly and accurately, providing:
        1. Clear defintions
        2. Relevant examples
        3. Policy implications
        4. Actionable insights
        Use accessible language suitable for policymakers without deep technical backgrounds.
        """
    
    def _handle_error(self, error):
        error_msg = str(error)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            logging.error(f"LLM Authentication Error: {error_msg}")
            return "Error: LLM authentication failed. Please check API key configuration."
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            logging.error(f"LLM Rate Limit Error: {error_msg}")
            return "Error: LLM rate limit exceeded. Please try again later."
        elif "timeout" in error_msg.lower():
            logging.error(f"LLM Timeout Error: {error_msg}")
            return "Error: LLM request timed out. Please try again later."
        else:
            logging.error(f"LLM Error: {error_msg}")
            return "Error: An unexpected error occurred while generating the response. Please try again later."
        
    