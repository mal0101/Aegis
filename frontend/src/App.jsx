import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './components/Home';
import ConceptChat from './components/ConceptChat';
import CaseStudyBrowser from './components/CaseStudyBrowser';
import ImpactSimulator from './components/ImpactSimulator';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/concepts" element={<ConceptChat />} />
        <Route path="/case-studies" element={<CaseStudyBrowser />} />
        <Route path="/simulator" element={<ImpactSimulator />} />
      </Routes>
    </Layout>
  );
}

export default App;
