import { useState, useEffect } from 'react';
import { Play, Loader2, ChevronDown, AlertTriangle, CheckCircle2, TrendingUp, BarChart3 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { getTemplates, predictImpact } from '../api';

function DimensionBar({ dimension }) {
  const score = dimension.score;
  const pct = Math.min(100, Math.max(0, score * 10));
  const color = score >= 7 ? 'bg-green-500' : score >= 4 ? 'bg-yellow-500' : 'bg-red-500';
  const confidenceColor = {
    high: 'text-green-600',
    medium: 'text-yellow-600',
    low: 'text-gray-400',
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <div className="flex justify-between items-center mb-2">
        <h4 className="font-semibold text-rich-black">{dimension.name}</h4>
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold text-rich-black">{score.toFixed(1)}</span>
          <span className="text-xs text-gray-400">/10</span>
          <span className={`text-xs ${confidenceColor[dimension.confidence] || 'text-gray-400'}`}>
            ({dimension.confidence})
          </span>
        </div>
      </div>
      <div className="w-full bg-gray-100 rounded-full h-2.5 mb-2">
        <div className={`${color} h-2.5 rounded-full transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
      <p className="text-sm text-gray-600">{dimension.explanation}</p>
    </div>
  );
}

function ImpactReport({ result }) {
  if (!result) return null;

  return (
    <div className="space-y-6">
      <div className="bg-violet-blue-50 rounded-xl p-5 border border-violet-blue-200">
        <h3 className="text-lg font-semibold text-violet-blue-900 mb-2">Executive Summary</h3>
        <p className="text-violet-blue-800">{result.executive_summary}</p>
        <div className="flex gap-4 mt-3 text-sm text-violet-blue">
          <span>Evidence base: {result.evidence_base_size} policies</span>
          <span>Processing: {(result.processing_time_ms / 1000).toFixed(1)}s</span>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-rich-black mb-3">Impact Dimensions</h3>
        <div className="grid md:grid-cols-2 gap-3">
          {result.impact_dimensions?.map((dim, i) => (
            <DimensionBar key={i} dimension={dim} />
          ))}
        </div>
      </div>

      {result.similar_policies?.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-rich-black mb-3">Similar International Policies</h3>
          <div className="flex flex-wrap gap-2">
            {result.similar_policies.map(sp => (
              <div key={sp.id} className="bg-white border border-gray-200 rounded-lg px-3 py-2 text-sm">
                <span className="font-medium text-rich-black">{sp.policy_name}</span>
                <span className="text-gray-400 ml-1">({sp.country})</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {result.recommendations?.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-rich-black mb-3">Recommendations</h3>
          <ul className="space-y-2">
            {result.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                <CheckCircle2 size={16} className="text-violet-blue mt-0.5 shrink-0" />
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}

      {result.full_analysis && (
        <details className="bg-white rounded-xl border border-gray-200 p-4">
          <summary className="cursor-pointer font-semibold text-rich-black flex items-center gap-2">
            <ChevronDown size={16} />
            Full Analysis
          </summary>
          <div className="prose prose-sm max-w-none mt-3 text-gray-700">
            <ReactMarkdown>{result.full_analysis}</ReactMarkdown>
          </div>
        </details>
      )}
    </div>
  );
}

export default function ImpactSimulator() {
  const [templates, setTemplates] = useState([]);
  const [form, setForm] = useState({
    policy_name: '',
    description: '',
    sectors: [],
    policy_type: 'comprehensive',
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const sectors = ['agriculture', 'tourism', 'manufacturing', 'public_services', 'healthcare', 'education', 'finance'];
  const policyTypes = ['comprehensive', 'sectoral', 'voluntary', 'sandbox', 'national_strategy', 'bill'];

  useEffect(() => {
    getTemplates().then(res => setTemplates(res.data)).catch(() => {});
  }, []);

  const loadTemplate = (template) => {
    setForm({
      policy_name: template.name,
      description: template.description,
      sectors: template.sectors,
      policy_type: template.policy_type,
    });
    setResult(null);
    setError('');
  };

  const toggleSector = (sector) => {
    setForm(prev => ({
      ...prev,
      sectors: prev.sectors.includes(sector)
        ? prev.sectors.filter(s => s !== sector)
        : [...prev.sectors, sector],
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.policy_name || !form.description) {
      setError('Please provide a policy name and description.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const res = await predictImpact(form);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Please check that the LLM service is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-rich-black">Impact Simulator</h1>
        <p className="text-gray-600 text-sm">Predict how a proposed AI policy would impact Morocco.</p>
      </div>

      {templates.length > 0 && (
        <div className="mb-6">
          <p className="text-sm text-gray-500 mb-2">Start from a template:</p>
          <div className="flex flex-wrap gap-2">
            {templates.map(t => (
              <button
                key={t.id}
                onClick={() => loadTemplate(t)}
                className="px-3 py-1.5 text-sm bg-white border border-gray-200 rounded-lg hover:border-violet-blue-200 hover:bg-violet-blue-50 transition-colors"
              >
                {t.name}
              </button>
            ))}
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="bg-white rounded-xl border border-gray-200 p-5 mb-6 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Policy Name</label>
          <input
            type="text"
            value={form.policy_name}
            onChange={e => setForm(prev => ({ ...prev, policy_name: e.target.value }))}
            placeholder="e.g., Morocco AI Transparency Act"
            className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-blue"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
          <textarea
            value={form.description}
            onChange={e => setForm(prev => ({ ...prev, description: e.target.value }))}
            placeholder="Describe the proposed policy..."
            rows={3}
            className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-blue"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Target Sectors</label>
          <div className="flex flex-wrap gap-2">
            {sectors.map(s => (
              <button
                key={s}
                type="button"
                onClick={() => toggleSector(s)}
                className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                  form.sectors.includes(s)
                    ? 'bg-violet-blue-50 border-violet-blue-200 text-violet-blue'
                    : 'bg-white border-gray-200 text-gray-600 hover:border-gray-300'
                }`}
              >
                {s.replace('_', ' ')}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Policy Type</label>
          <select
            value={form.policy_type}
            onChange={e => setForm(prev => ({ ...prev, policy_type: e.target.value }))}
            className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-blue"
          >
            {policyTypes.map(pt => (
              <option key={pt} value={pt}>{pt.replace('_', ' ')}</option>
            ))}
          </select>
        </div>

        {error && (
          <div className="flex items-center gap-2 text-red-600 text-sm">
            <AlertTriangle size={16} />
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 bg-violet-blue text-white rounded-xl hover:bg-violet-blue-700 disabled:opacity-50 font-medium flex items-center justify-center gap-2 transition-colors"
        >
          {loading ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              Running simulation...
            </>
          ) : (
            <>
              <Play size={18} />
              Run Impact Simulation
            </>
          )}
        </button>
      </form>

      <ImpactReport result={result} />
    </div>
  );
}
