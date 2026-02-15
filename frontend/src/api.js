import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
});

// Concepts
export const listConcepts = () => api.get('/api/concepts/list');
export const getConcept = (id) => api.get(`/api/concepts/${id}`);
export const askConcept = (question, difficulty) =>
  api.post('/api/concepts/ask', { question, difficulty });

// Case Studies
export const listCaseStudies = () => api.get('/api/case-studies/');
export const getCaseStudy = (id) => api.get(`/api/case-studies/${id}`);
export const searchCaseStudies = (query, country, policyType, topK = 5) =>
  api.post('/api/case-studies/search', { query, country, policy_type: policyType, top_k: topK });
export const compareCaseStudies = (ids) =>
  api.post('/api/case-studies/compare', { ids });

// Simulator
export const getTemplates = () => api.get('/api/simulate/templates');
export const predictImpact = (proposal) =>
  api.post('/api/simulate/predict', proposal);

// Health
export const healthCheck = () => api.get('/health');

export default api;
