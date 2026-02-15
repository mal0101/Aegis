import { Link } from 'react-router-dom';
import { Brain, BookOpen, BarChart3, ArrowRight, CheckCircle2, Lock, Shield } from 'lucide-react';

const features = [
  {
    title: 'AI Concept Simulator',
    description: 'Model policy outcomes in a secure local environment with custom RAG pipelines tailored for Moroccan demographics.',
    icon: Brain,
    path: '/concepts',
  },
  {
    title: 'Case Study Library',
    description: 'Access a repository of global and regional policy frameworks, benchmarks, and historical legislative data.',
    icon: BookOpen,
    path: '/case-studies',
  },
  {
    title: 'Impact Simulator',
    description: 'Predict socioeconomic shifts with precision AI modeling using local datasets without cloud dependency.',
    icon: BarChart3,
    path: '/simulator',
  },
];

const standardPoints = [
  'GDPR & Local Data Law Compliant',
  'Multi-ministerial Integration Capable',
  'Real-time Socio-Economic Analysis',
];

export default function Home() {
  return (
    <div className="space-y-24">
      {/* ── Hero Section ── */}
      <section className="text-center pt-12 pb-4">
        {/* Live badge */}
        <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white border border-gray-200 rounded-full text-xs font-medium text-gray-600 mb-8">
          <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          NOW LIVE: LLAMA 3.3 70B OPTIMIZED
        </div>

        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-rich-black leading-tight mb-6">
          AI-Powered Decision Support
          <br />
          for <span className="italic text-violet-blue">Morocco&rsquo;s Future</span>
        </h1>

        <p className="text-gray-500 max-w-xl mx-auto mb-8 leading-relaxed">
          Leveraging RAG-powered local simulations with Llama 3.3 70B for
          secure, data-driven policy modeling. Built for the sovereignty
          of national data.
        </p>

        <div className="flex items-center justify-center gap-4">
          <Link
            to="/simulator"
            className="inline-flex items-center gap-2 px-6 py-3 bg-violet-blue text-white text-sm font-semibold rounded-lg hover:bg-violet-blue-700 transition-colors"
          >
            Start Simulation <ArrowRight size={16} />
          </Link>
          <Link
            to="/case-studies"
            className="inline-flex items-center gap-2 px-6 py-3 bg-white text-rich-black text-sm font-semibold rounded-lg border border-gray-300 hover:border-violet-blue-200 hover:bg-gray-50 transition-colors"
          >
            Browse Library
          </Link>
        </div>
      </section>

      {/* ── Advanced Policy Tools ── */}
      <section>
        <div className="mb-8">
          <h2 className="text-2xl sm:text-3xl font-bold text-violet-blue mb-2">Advanced Policy Tools</h2>
          <p className="text-gray-500 max-w-lg text-sm">
            Designed for precision and absolute data sovereignty in governmental decision-making processes.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-5">
          {features.map(({ title, description, icon: Icon, path }) => (
            <Link
              key={path}
              to={path}
              className="block p-6 bg-white rounded-xl border border-gray-200 hover:border-violet-blue-200 hover:shadow-md transition-all group"
            >
              <div className="w-11 h-11 bg-violet-blue rounded-lg flex items-center justify-center mb-5">
                <Icon size={20} className="text-white" />
              </div>
              <h3 className="text-base font-bold text-rich-black mb-2">{title}</h3>
              <p className="text-gray-500 text-sm leading-relaxed">{description}</p>
            </Link>
          ))}
        </div>
      </section>

      {/* ── The Aegis Standard ── */}
      <section>
        <div className="flex flex-col lg:flex-row gap-10 lg:gap-16 items-start">
          {/* Left column */}
          <div className="flex-1">
            <h2 className="text-2xl sm:text-3xl font-bold text-rich-black mb-4">The Aegis Standard</h2>
            <p className="text-gray-500 text-sm leading-relaxed mb-6">
              Our platform is architected for the unique needs of the Moroccan
              administration. By deploying LLMs locally, we ensure that sensitive
              policy drafts and citizen data never leave national borders.
            </p>
            <ul className="space-y-3">
              {standardPoints.map(point => (
                <li key={point} className="flex items-center gap-3 text-sm text-rich-black font-medium">
                  <CheckCircle2 size={18} className="text-violet-blue shrink-0" />
                  {point}
                </li>
              ))}
            </ul>
          </div>

          {/* Right column — stat cards */}
          <div className="flex flex-col sm:flex-row lg:flex-col xl:flex-row gap-4 shrink-0">
            {/* Benchmarks card */}
            <div className="bg-white border border-gray-200 rounded-xl p-6 w-52">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Benchmarks</p>
              <p className="text-4xl font-bold text-rich-black mb-1">8+</p>
              <p className="text-sm text-gray-500 mb-3">Global Policies Integrated</p>
              <div className="w-12 h-1 bg-violet-blue rounded-full" />
            </div>

            {/* Sovereignty card */}
            <div className="bg-rich-black rounded-xl p-6 w-52 text-white">
              <p className="text-xs font-semibold text-violet-blue-200 uppercase tracking-wider mb-3">Sovereignty</p>
              <p className="text-3xl font-bold mb-0.5">100%</p>
              <p className="text-lg font-bold italic mb-1">Offline</p>
              <p className="text-sm text-gray-400 mb-4">Data-Secure Local Deployment</p>
              <div className="flex items-center gap-1.5 text-xs text-gray-400">
                <Lock size={12} className="text-violet-blue-200" />
                <span className="font-semibold uppercase tracking-wide">Military Grade Encryption</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
