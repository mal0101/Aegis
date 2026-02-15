import { Link, useLocation } from 'react-router-dom';
import { Brain, BookOpen, BarChart3, Home, Shield } from 'lucide-react';

const navItems = [
  { path: '/', label: 'Home', icon: Home },
  { path: '/concepts', label: 'Concepts', icon: Brain },
  { path: '/case-studies', label: 'Case Studies', icon: BookOpen },
  { path: '/simulator', label: 'Simulator', icon: BarChart3 },
];

const footerLinks = {
  Platform: [
    { label: 'Simulations', to: '/simulator' },
    { label: 'Library', to: '/case-studies' },
    { label: 'Benchmarks', to: '/case-studies' },
  ],
  Governance: [
    { label: 'Security', to: '/' },
    { label: 'Data Privacy', to: '/' },
    { label: 'Compliance', to: '/' },
  ],
  Support: [
    { label: 'Documentation', to: '/' },
    { label: 'API', to: '/' },
    { label: 'Contact', to: '/' },
  ],
};

export default function Layout({ children }) {
  const location = useLocation();

  return (
    <div className="min-h-screen flex flex-col bg-alice-blue">
      <nav className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <Link to="/" className="flex items-center gap-2">
                <div className="w-8 h-8 bg-violet-blue rounded-lg flex items-center justify-center">
                  <Shield size={18} className="text-white" />
                </div>
                <span className="text-xl font-bold text-rich-black">Aegis</span>
              </Link>
            </div>
            <div className="flex items-center gap-1">
              {navItems.map(({ path, label, icon: Icon }) => (
                <Link
                  key={path}
                  to={path}
                  className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    location.pathname === path
                      ? 'bg-violet-blue-50 text-violet-blue'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  }`}
                >
                  <Icon size={16} />
                  <span className="hidden sm:inline">{label}</span>
                </Link>
              ))}
            </div>
          </div>
        </div>
      </nav>

      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
        {children}
      </main>

      <footer className="bg-rich-black text-white mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-10">
            {/* Brand column */}
            <div className="lg:col-span-2">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-7 h-7 bg-violet-blue rounded-lg flex items-center justify-center">
                  <Shield size={14} className="text-white" />
                </div>
                <span className="text-lg font-bold">Aegis</span>
              </div>
              <p className="text-sm text-gray-400 leading-relaxed max-w-xs">
                Bridging the gap between data and policy through secure, local AI simulations.
                Optimized for the Kingdom of Morocco.
              </p>
            </div>

            {/* Link columns */}
            {Object.entries(footerLinks).map(([heading, links]) => (
              <div key={heading}>
                <h4 className="text-xs font-bold uppercase tracking-wider text-gray-400 mb-4">{heading}</h4>
                <ul className="space-y-2.5">
                  {links.map(link => (
                    <li key={link.label}>
                      <Link to={link.to} className="text-sm text-gray-400 hover:text-white transition-colors">
                        {link.label}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          {/* Bottom bar */}
          <div className="mt-10 pt-6 border-t border-gray-800 flex flex-col sm:flex-row justify-between items-center gap-3">
            <p className="text-xs text-gray-500">&copy; 2026 Aegis. All rights reserved.</p>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <span className="uppercase tracking-wider font-semibold">Made for Morocco</span>
              <span className="text-xl leading-none">&#x1F1F2;&#x1F1E6;</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
