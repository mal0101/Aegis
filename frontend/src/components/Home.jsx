import { Link } from 'react-router-dom';
import { Brain, BookOpen, BarChart3, ArrowRight } from 'lucide-react';

const features = [
  {
    title: 'AI Concept Simulator',
    description: 'Ask questions about AI policy concepts and get answers tailored to Morocco\'s context, legal framework, and development goals.',
    icon: Brain,
    path: '/concepts',
    color: 'emerald',
  },
  {
    title: 'Case Study Library',
    description: 'Explore 8 international AI policies with real outcome data. Compare approaches from the EU, Singapore, Rwanda, and more.',
    icon: BookOpen,
    path: '/case-studies',
    color: 'blue',
  },
  {
    title: 'Impact Simulator',
    description: 'Predict the multi-dimensional impact of proposed AI policies on Morocco using evidence from similar international cases.',
    icon: BarChart3,
    path: '/simulator',
    color: 'purple',
  },
];

const colorMap = {
  emerald: { bg: 'bg-emerald-50', icon: 'bg-emerald-600', text: 'text-emerald-700', hover: 'hover:border-emerald-300' },
  blue: { bg: 'bg-blue-50', icon: 'bg-blue-600', text: 'text-blue-700', hover: 'hover:border-blue-300' },
  purple: { bg: 'bg-purple-50', icon: 'bg-purple-600', text: 'text-purple-700', hover: 'hover:border-purple-300' },
};

export default function Home() {
  return (
    <div>
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">PolicyBridge</h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          AI-powered decision support for Moroccan policymakers. Understand AI concepts,
          explore international case studies, and simulate policy impacts — all locally, for free.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {features.map(({ title, description, icon: Icon, path, color }) => {
          const c = colorMap[color];
          return (
            <Link
              key={path}
              to={path}
              className={`block p-6 bg-white rounded-xl border border-gray-200 ${c.hover} transition-all hover:shadow-md`}
            >
              <div className={`w-12 h-12 ${c.icon} rounded-lg flex items-center justify-center mb-4`}>
                <Icon size={24} className="text-white" />
              </div>
              <h2 className="text-xl font-semibold text-gray-900 mb-2">{title}</h2>
              <p className="text-gray-600 text-sm mb-4">{description}</p>
              <div className={`flex items-center gap-1 ${c.text} text-sm font-medium`}>
                Get started <ArrowRight size={14} />
              </div>
            </Link>
          );
        })}
      </div>

      <div className="mt-12 p-6 bg-white rounded-xl border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">About PolicyBridge</h3>
        <p className="text-gray-600 text-sm">
          Built for the Morocco AI policy hackathon. PolicyBridge combines RAG-powered concept explanation,
          a searchable library of international AI policies with real metrics, and an evidence-based impact
          simulator. Everything runs locally using Ollama + Mistral 7B — zero cost, zero API keys, full privacy.
        </p>
      </div>
    </div>
  );
}
