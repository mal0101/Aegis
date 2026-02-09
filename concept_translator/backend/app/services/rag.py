from typing import Dict, List, Any, Optional
import logging
import time
import json
from pathlib import Path
from concept_translator.backend.app.services.vector_db import VectorDBService
from concept_translator.backend.app.core.config import settings
from concept_translator.backend.app.services.llm import LLMService

logger = logging.getLogger(__name__)
class RAGService:
    def __init__(self):
        try:
            self.vector_db = VectorDBService()
            self.llm = LLMService()
            self.concepts_data = self._load_concepts_data()
            self.concepts_by_id = {concept["id"]: concept for concept in self.concepts_data}
            
        except Exception as e:
            logger.error(f"Error initializing RAGService: {str(e)}")
            raise RuntimeError(f"Failed to initialize RAGService: {str(e)}")
    
    def answer_query(self, query, n_results=3, include_metadata=True):
        if not query or not query.strip():
            return
        start_time = time.time()
        try:
            search_results = self.retrieve(query, n_results)
            concept_ids = search_results['ids'][0] if search_results['ids'] else []
            if not concept_ids:
                return self._no_results_response(query)
            concepts = self._load_concepts_data(concept_ids)
            if not concepts:
                return self._no_results_response(query)
            context = self._build_context(concepts)
            answer = self.llm.generate_with_context(
                question=query,
                context=context,
                include_morocco_focus=True
            )
            metadata = {}
            if include_metadata:
                metadata = self._extract_metadata(concepts)
            processing_time = time.time() - start_time
            response = {
                "answer" : answer,
                "related_concepts": metadata.get('related_concepts', []),
                "sources": metadata.get('sources', []),
                "retrieved_concepts": concept_ids,
                "processing_time": round(processing_time, 2)
            }
            return response
        except Exception as e:
            return self._error_response(query, str(e))
    
    def retrieve(self, query, n_results):
        return self.vector_db.search(query, n_results=n_results)

    def _load_concepts_data(self, concept_ids):
        concepts = []
        for concept_id in concept_ids:
            if concept_id in self.concepts_by_id:
                concepts.append(self.concepts_by_id[concept_id])
            else:
                logger.warning(f"Concept ID {concept_id} not found in concepts data")
        return concepts
    
    def _build_context(self, concepts):
        context_parts = []
        for i, concept in enumerate(concepts,1):
            concept_text = f"""
            CONCEPT {i}: {concept.get('term','Unknown')}
            Definition:
            {concept.get('definition', 'N/A')}
            Simple Explanation:
            {concept.get('simple_explanation', 'N/A')}
            Examples:
            {self._format_examples(concept.get('examples', []))}
            Morocco Policy Relevance:
            {concept.get('policy_relevance','N/A')}
            Categories: {', '.join(concept.get('categories',[]))}
            Difficulty: {concept.get('difficulty_level', 'N/A')}
            """
            context_parts.append(concept_text)
            full_context = "\n\n".join(context_parts)
            return full_context
    
    def _extract_metadata(self, concepts):
        pass
    
    def _error_response(self, query, error_message):
        pass
    
    def _no_results_response(self, query):
        pass
    
    def _load_concepts(self):
        pass
    