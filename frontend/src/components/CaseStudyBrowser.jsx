import { useState, useEffect } from 'react';
import { Search, X, Globe, Calendar, Tag, BarChart2, Loader2 } from 'lucide-react';
import { listCaseStudies, searchCaseStudies, getCaseStudy } from '../api';

function CaseStudyCard({ study, onClick }) {
  const typeColors = {
    comprehensive: 'bg-blue-100 text-blue-700',
    voluntary: 'bg-green-100 text-green-700',
    national_strategy: 'bg-purple-100 text-purple-700',
    bill: 'bg-orange-100 text-orange-700',
    sandbox: 'bg-yellow-100 text-yellow-700',
  };

  return (
    <div
      onClick={onClick}
      className="bg-white rounded-xl border border-gray-200 p-5 hover:shadow-md hover:border-blue-300 transition-all cursor-pointer"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <Globe size={16} className="text-gray-400" />
          <span className="font-semibold text-gray-900">{study.country}</span>
        </div>
        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${typeColors[study.policy_type] || 'bg-gray-100 text-gray-600'}`}>
          {study.policy_type?.replace('_', ' ')}
        </span>
      </div>
      <h3 className="text-lg font-semibold text-gray-900 mb-2">{study.policy_name}</h3>
      <div className="flex items-center gap-4 text-sm text-gray-500">
        <span className="flex items-center gap-1">
          <Calendar size={14} />
          {study.enacted_date}
        </span>
        <span className={`flex items-center gap-1 ${
          study.data_quality === 'high' ? 'text-green-600' :
          study.data_quality === 'medium' ? 'text-yellow-600' : 'text-gray-400'
        }`}>
          <BarChart2 size={14} />
          {study.data_quality} quality
        </span>
      </div>
      {study.tags && study.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-3">
          {study.tags.map(tag => (
            <span key={tag} className="px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded-full">
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function CaseStudyDetail({ study, onClose }) {
  if (!study) return null;

  const social = study.outcomes?.social_impact || {};
  const economic = study.outcomes?.economic_impact || {};
  const impl = study.outcomes?.implementation_reality || {};

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div className="bg-white rounded-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto p-6" onClick={e => e.stopPropagation()}>
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">{study.policy?.name}</h2>
            <p className="text-gray-500">{study.country} — {study.policy?.enacted_date}</p>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded-lg">
            <X size={20} />
          </button>
        </div>

        <p className="text-gray-700 mb-6">{study.policy?.description}</p>

        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div className="bg-blue-50 rounded-xl p-4">
            <h4 className="text-sm font-semibold text-blue-800 mb-2">Social Impact</h4>
            <div className="space-y-1">
              <p className="text-sm">Trust change: <span className="font-semibold text-blue-700">+{social.trust_change_pct}%</span></p>
              <p className="text-sm">Bias reduction: <span className="font-semibold text-blue-700">{social.bias_reduction_pct}%</span></p>
            </div>
          </div>
          <div className="bg-green-50 rounded-xl p-4">
            <h4 className="text-sm font-semibold text-green-800 mb-2">Economic Impact</h4>
            <div className="space-y-1">
              <p className="text-sm">Compliance cost: <span className="font-semibold text-green-700">${economic.compliance_costs_usd?.toLocaleString()}</span></p>
              <p className="text-sm">Startup growth: <span className={`font-semibold ${economic.startup_growth_pct >= 0 ? 'text-green-700' : 'text-red-600'}`}>{economic.startup_growth_pct}%</span></p>
            </div>
          </div>
          <div className="bg-purple-50 rounded-xl p-4">
            <h4 className="text-sm font-semibold text-purple-800 mb-2">Implementation</h4>
            <div className="space-y-1">
              <p className="text-sm">Timeline: <span className="font-semibold text-purple-700">{impl.timeline_months} months</span></p>
              <p className="text-sm">Compliance: <span className="font-semibold text-purple-700">{impl.compliance_rate_pct}%</span></p>
            </div>
          </div>
        </div>

        {study.policy?.key_provisions && (
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-gray-900 mb-2">Key Provisions</h4>
            <ul className="space-y-1">
              {study.policy.key_provisions.map((p, i) => (
                <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                  <span className="text-emerald-500 mt-1">-</span> {p}
                </li>
              ))}
            </ul>
          </div>
        )}

        {study.outcomes?.qualitative_insights && (
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-gray-900 mb-2">Lessons Learned</h4>
            <ul className="space-y-2">
              {study.outcomes.qualitative_insights.map((insight, i) => (
                <li key={i} className="text-sm text-gray-600 bg-gray-50 rounded-lg p-3">
                  {insight}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="flex gap-4 text-xs text-gray-400">
          <span>GDP ratio to Morocco: {study.metadata?.gdp_ratio_to_morocco}x</span>
          <span>Legal similarity: {(study.metadata?.legal_similarity * 100)?.toFixed(0)}%</span>
          <span>Tech gap: {(study.metadata?.tech_maturity_gap * 100)?.toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );
}

export default function CaseStudyBrowser() {
  const [studies, setStudies] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [searching, setSearching] = useState(false);
  const [selected, setSelected] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    listCaseStudies()
      .then(res => setStudies(res.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      listCaseStudies().then(res => setStudies(res.data));
      return;
    }
    setSearching(true);
    try {
      const res = await searchCaseStudies(searchQuery);
      setStudies(res.data);
    } catch (err) {
      // keep existing results
    } finally {
      setSearching(false);
    }
  };

  const handleCardClick = async (studyId) => {
    setDetailLoading(true);
    try {
      const res = await getCaseStudy(studyId);
      setSelected(res.data);
    } catch (err) {
      // ignore
    } finally {
      setDetailLoading(false);
    }
  };

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Case Study Library</h1>
        <p className="text-gray-600 text-sm">Explore international AI policies with real outcome data.</p>
      </div>

      <form onSubmit={handleSearch} className="flex gap-2 mb-6">
        <div className="relative flex-1">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search policies (e.g., 'transparency regulation Africa')..."
            className="w-full pl-10 pr-4 py-2.5 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <button
          type="submit"
          disabled={searching}
          className="px-4 py-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {searching ? <Loader2 size={16} className="animate-spin" /> : 'Search'}
        </button>
      </form>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 size={24} className="animate-spin text-blue-500" />
        </div>
      ) : (
        <div className="grid md:grid-cols-2 gap-4">
          {studies.map(study => (
            <CaseStudyCard key={study.id} study={study} onClick={() => handleCardClick(study.id)} />
          ))}
        </div>
      )}

      {studies.length === 0 && !loading && (
        <p className="text-center text-gray-500 py-12">No case studies found.</p>
      )}

      {selected && <CaseStudyDetail study={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
