import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Search, X, Loader2, ChevronRight, Download, ArrowRight,
  CheckCircle2, Info, AlertTriangle, RotateCcw, Globe, Calendar, BarChart2
} from 'lucide-react';
import { listCaseStudies, getCaseStudy } from '../api';

const ITEMS_PER_PAGE = 6;

const REGION_MAP = {
  'Europe': ['EU', 'UK'],
  'North America': ['Canada'],
  'Asia Pacific': ['Singapore', 'South Korea'],
  'Africa': ['Rwanda', 'Tunisia'],
  'Latin America': ['Brazil'],
};

const COUNTRY_FLAGS = {
  'EU': '\u{1F1EA}\u{1F1FA}',
  'Canada': '\u{1F1E8}\u{1F1E6}',
  'Singapore': '\u{1F1F8}\u{1F1EC}',
  'Rwanda': '\u{1F1F7}\u{1F1FC}',
  'Brazil': '\u{1F1E7}\u{1F1F7}',
  'UK': '\u{1F1EC}\u{1F1E7}',
  'Tunisia': '\u{1F1F9}\u{1F1F3}',
  'South Korea': '\u{1F1F0}\u{1F1F7}',
};

const TYPE_BADGE_COLORS = {
  'comprehensive': 'bg-violet-blue-50 text-violet-blue border border-violet-blue-200',
  'bill': 'bg-yellow-50 text-yellow-700 border border-yellow-200',
  'voluntary': 'bg-green-50 text-green-700 border border-green-200',
  'national_strategy': 'bg-blue-50 text-blue-700 border border-blue-200',
  'sandbox': 'bg-orange-50 text-orange-700 border border-orange-200',
};

const QUALITY_LABELS = {
  'high': { label: 'High Quality', color: 'text-green-600', icon: CheckCircle2 },
  'medium': { label: 'Standardized', color: 'text-blue-600', icon: Info },
  'projected': { label: 'Projected', color: 'text-yellow-600', icon: AlertTriangle },
};

function CaseStudyCard({ study, onClick }) {
  const badgeColor = TYPE_BADGE_COLORS[study.policy_type] || 'bg-gray-100 text-gray-600 border border-gray-200';
  const quality = QUALITY_LABELS[study.data_quality] || QUALITY_LABELS['medium'];
  const QualityIcon = quality.icon;
  const flag = COUNTRY_FLAGS[study.country] || '';

  return (
    <div
      onClick={onClick}
      className="bg-white rounded-xl border border-gray-200 p-5 hover:shadow-md hover:border-violet-blue-200 transition-all cursor-pointer flex flex-col"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-lg">{flag}</span>
          <span className="font-semibold text-rich-black text-sm">{study.country}</span>
        </div>
        <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium uppercase ${badgeColor}`}>
          {study.policy_type?.replace('_', ' ')}
        </span>
      </div>
      <h3 className="text-base font-semibold text-rich-black mb-2 line-clamp-2">{study.policy_name}</h3>
      <p className="text-xs text-gray-500 mb-3">
        Published: {study.enacted_date}
      </p>
      {study.tags && study.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3">
          {study.tags.slice(0, 3).map(tag => (
            <span key={tag} className="px-2 py-0.5 text-xs bg-alice-blue text-gray-600 rounded-full">
              {tag}
            </span>
          ))}
        </div>
      )}
      <div className="mt-auto pt-3 border-t border-gray-100 flex items-center justify-between">
        <div className={`flex items-center gap-1 text-xs ${quality.color}`}>
          <QualityIcon size={12} />
          <span>{quality.label}</span>
        </div>
        <span className="text-xs font-medium text-violet-blue flex items-center gap-1">
          DETAILS <ArrowRight size={12} />
        </span>
      </div>
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
            <h2 className="text-2xl font-bold text-rich-black">{study.policy?.name}</h2>
            <p className="text-gray-500">{study.country} — {study.policy?.enacted_date}</p>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded-lg">
            <X size={20} />
          </button>
        </div>

        <p className="text-gray-700 mb-6">{study.policy?.description}</p>

        <div className="grid md:grid-cols-3 gap-4 mb-6">
          <div className="bg-violet-blue-50 rounded-xl p-4">
            <h4 className="text-sm font-semibold text-violet-blue-800 mb-2">Social Impact</h4>
            <div className="space-y-1">
              <p className="text-sm">Trust change: <span className="font-semibold text-violet-blue">+{social.trust_change_pct}%</span></p>
              <p className="text-sm">Bias reduction: <span className="font-semibold text-violet-blue">{social.bias_reduction_pct}%</span></p>
            </div>
          </div>
          <div className="bg-green-50 rounded-xl p-4">
            <h4 className="text-sm font-semibold text-green-800 mb-2">Economic Impact</h4>
            <div className="space-y-1">
              <p className="text-sm">Compliance cost: <span className="font-semibold text-green-700">${economic.compliance_costs_usd?.toLocaleString()}</span></p>
              <p className="text-sm">Startup growth: <span className={`font-semibold ${economic.startup_growth_pct >= 0 ? 'text-green-700' : 'text-red-600'}`}>{economic.startup_growth_pct}%</span></p>
            </div>
          </div>
          <div className="bg-eggshell-100 rounded-xl p-4">
            <h4 className="text-sm font-semibold text-rich-black mb-2">Implementation</h4>
            <div className="space-y-1">
              <p className="text-sm">Timeline: <span className="font-semibold text-rich-black">{impl.timeline_months} months</span></p>
              <p className="text-sm">Compliance: <span className="font-semibold text-rich-black">{impl.compliance_rate_pct}%</span></p>
            </div>
          </div>
        </div>

        {study.policy?.key_provisions && (
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-rich-black mb-2">Key Provisions</h4>
            <ul className="space-y-1">
              {study.policy.key_provisions.map((p, i) => (
                <li key={i} className="text-sm text-gray-600 flex items-start gap-2">
                  <span className="text-violet-blue mt-1">-</span> {p}
                </li>
              ))}
            </ul>
          </div>
        )}

        {study.outcomes?.qualitative_insights && (
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-rich-black mb-2">Lessons Learned</h4>
            <ul className="space-y-2">
              {study.outcomes.qualitative_insights.map((insight, i) => (
                <li key={i} className="text-sm text-gray-600 bg-alice-blue rounded-lg p-3">
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
  const [filteredStudies, setFilteredStudies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const [keyword, setKeyword] = useState('');
  const [selectedRegions, setSelectedRegions] = useState([]);
  const [selectedQualities, setSelectedQualities] = useState([]);
  const [sortBy, setSortBy] = useState('date');
  const [currentPage, setCurrentPage] = useState(1);

  useEffect(() => {
    listCaseStudies()
      .then(res => setStudies(res.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    let result = [...studies];

    if (keyword.trim()) {
      const kw = keyword.toLowerCase();
      result = result.filter(s =>
        s.policy_name.toLowerCase().includes(kw) ||
        s.country.toLowerCase().includes(kw) ||
        (s.tags && s.tags.some(t => t.toLowerCase().includes(kw)))
      );
    }

    if (selectedRegions.length > 0) {
      const allowedCountries = selectedRegions.flatMap(r => REGION_MAP[r] || []);
      result = result.filter(s => allowedCountries.includes(s.country));
    }

    if (selectedQualities.length > 0) {
      result = result.filter(s => selectedQualities.includes(s.data_quality));
    }

    if (sortBy === 'date') {
      result.sort((a, b) => new Date(b.enacted_date) - new Date(a.enacted_date));
    } else if (sortBy === 'country') {
      result.sort((a, b) => a.country.localeCompare(b.country));
    } else if (sortBy === 'name') {
      result.sort((a, b) => a.policy_name.localeCompare(b.policy_name));
    }

    setFilteredStudies(result);
    setCurrentPage(1);
  }, [studies, keyword, selectedRegions, selectedQualities, sortBy]);

  const totalPages = Math.ceil(filteredStudies.length / ITEMS_PER_PAGE);
  const paginatedStudies = filteredStudies.slice(
    (currentPage - 1) * ITEMS_PER_PAGE,
    currentPage * ITEMS_PER_PAGE
  );

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

  const resetFilters = () => {
    setKeyword('');
    setSelectedRegions([]);
    setSelectedQualities([]);
    setSortBy('date');
  };

  const toggleRegion = (region) => {
    setSelectedRegions(prev =>
      prev.includes(region) ? prev.filter(r => r !== region) : [...prev, region]
    );
  };

  const toggleQuality = (quality) => {
    setSelectedQualities(prev =>
      prev.includes(quality) ? prev.filter(q => q !== quality) : [...prev, quality]
    );
  };

  return (
    <div>
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-gray-500 mb-4">
        <Link to="/" className="hover:text-violet-blue transition-colors">Home</Link>
        <ChevronRight size={14} />
        <span className="text-rich-black font-medium">Case Study Library</span>
      </div>

      {/* Header row */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
        <div>
          <h1 className="text-3xl font-bold text-rich-black">AI Policy Repository</h1>
          <p className="text-gray-500 text-sm mt-1">Navigate a comprehensive database of global AI policy frameworks and validated international case studies.</p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={sortBy}
            onChange={e => setSortBy(e.target.value)}
            className="px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-violet-blue bg-white"
          >
            <option value="date">Sort by: Recent</option>
            <option value="country">Sort by: Country</option>
            <option value="name">Sort by: Name</option>
          </select>
          <button className="px-4 py-2 border border-gray-200 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 flex items-center gap-2 transition-colors">
            <Download size={14} />
            Export List
          </button>
        </div>
      </div>

      {/* Stats bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        {[
          { label: 'TOTAL STUDIES', value: studies.length },
          { label: 'GLOBAL REGIONS', value: Object.keys(REGION_MAP).length },
          { label: 'COMPREHENSIVE ACTS', value: studies.filter(s => s.policy_type === 'comprehensive').length },
          { label: 'HIGH QUALITY DATA', value: studies.filter(s => s.data_quality === 'high').length },
        ].map(stat => (
          <div key={stat.label} className="bg-white rounded-xl border border-gray-200 p-4">
            <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">{stat.label}</p>
            <p className="text-2xl font-bold text-violet-blue">{stat.value}</p>
          </div>
        ))}
      </div>

      {/* Main content: sidebar + grid */}
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Sidebar */}
        <aside className="w-full lg:w-64 shrink-0">
          <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-5 lg:sticky lg:top-24">
            <div>
              <h3 className="text-sm font-bold text-rich-black mb-1">Filters</h3>
              <p className="text-xs text-gray-400 uppercase tracking-wide">Refine Repository</p>
            </div>

            {/* Keywords */}
            <div>
              <label className="block text-sm font-semibold text-rich-black mb-2">Keywords</label>
              <div className="relative">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  value={keyword}
                  onChange={e => setKeyword(e.target.value)}
                  placeholder="e.g. EU Act, AIDA..."
                  className="w-full pl-9 pr-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-violet-blue"
                />
              </div>
            </div>

            {/* Region */}
            <div>
              <label className="block text-sm font-semibold text-rich-black mb-2">Region</label>
              <div className="space-y-2">
                {Object.keys(REGION_MAP).map(region => (
                  <label key={region} className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedRegions.includes(region)}
                      onChange={() => toggleRegion(region)}
                      className="rounded border-gray-300 text-violet-blue focus:ring-violet-blue accent-violet-blue"
                    />
                    {region}
                  </label>
                ))}
              </div>
            </div>

            {/* Data Quality */}
            <div>
              <label className="block text-sm font-semibold text-rich-black mb-2">Impact Quality</label>
              <div className="space-y-2">
                {Object.entries(QUALITY_LABELS).map(([key, { label }]) => (
                  <label key={key} className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedQualities.includes(key)}
                      onChange={() => toggleQuality(key)}
                      className="rounded border-gray-300 text-violet-blue focus:ring-violet-blue accent-violet-blue"
                    />
                    {label}
                  </label>
                ))}
              </div>
            </div>

            {/* Reset */}
            <button
              onClick={resetFilters}
              className="w-full py-2 text-sm text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors flex items-center justify-center gap-2"
            >
              <RotateCcw size={14} />
              Reset Filters
            </button>
          </div>
        </aside>

        {/* Card grid */}
        <div className="flex-1">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 size={24} className="animate-spin text-violet-blue" />
            </div>
          ) : paginatedStudies.length > 0 ? (
            <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-4">
              {paginatedStudies.map(study => (
                <CaseStudyCard key={study.id} study={study} onClick={() => handleCardClick(study.id)} />
              ))}
            </div>
          ) : (
            <p className="text-center text-gray-500 py-12">No case studies match your filters.</p>
          )}

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex justify-center items-center gap-2 mt-6">
              <button
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1.5 text-sm border border-gray-200 rounded-lg disabled:opacity-40 hover:bg-gray-50 transition-colors"
              >
                &lsaquo;
              </button>
              {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
                <button
                  key={page}
                  onClick={() => setCurrentPage(page)}
                  className={`w-9 h-9 text-sm rounded-lg transition-colors ${
                    page === currentPage
                      ? 'bg-violet-blue text-white'
                      : 'border border-gray-200 hover:bg-gray-50'
                  }`}
                >
                  {page}
                </button>
              ))}
              <button
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1.5 text-sm border border-gray-200 rounded-lg disabled:opacity-40 hover:bg-gray-50 transition-colors"
              >
                &rsaquo;
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Detail modal */}
      {selected && <CaseStudyDetail study={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
