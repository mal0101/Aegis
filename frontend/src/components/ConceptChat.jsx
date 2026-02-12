import { useState, useEffect, useRef } from 'react';
import { Send, Loader2, BookOpen } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { askConcept, listConcepts } from '../api';

export default function ConceptChat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [concepts, setConcepts] = useState([]);
  const messagesEnd = useRef(null);

  useEffect(() => {
    listConcepts().then(res => setConcepts(res.data)).catch(() => {});
  }, []);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: question }]);
    setLoading(true);

    try {
      const res = await askConcept(question);
      const data = res.data;
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        related: data.related_concepts,
        sources: data.sources,
        time: data.processing_time_ms,
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I could not process your question. Make sure Ollama is running with Mistral.',
        error: true,
      }]);
    } finally {
      setLoading(false);
    }
  };

  const quickAsk = (term) => {
    setInput(`What is ${term} and why does it matter for Morocco?`);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-10rem)]">
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-gray-900">AI Concept Simulator</h1>
        <p className="text-gray-600 text-sm">Ask about AI policy concepts — answers are tailored to Morocco's context.</p>
      </div>

      {concepts.length > 0 && messages.length === 0 && (
        <div className="mb-4">
          <p className="text-xs text-gray-500 mb-2">Quick questions:</p>
          <div className="flex flex-wrap gap-2">
            {concepts.slice(0, 6).map(c => (
              <button
                key={c.id}
                onClick={() => quickAsk(c.term)}
                className="px-3 py-1.5 text-xs bg-white border border-gray-200 rounded-full hover:border-emerald-300 hover:bg-emerald-50 transition-colors"
              >
                {c.term}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto space-y-4 mb-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-xl px-4 py-3 ${
              msg.role === 'user'
                ? 'bg-emerald-600 text-white'
                : msg.error
                  ? 'bg-red-50 border border-red-200 text-red-700'
                  : 'bg-white border border-gray-200 text-gray-800'
            }`}>
              {msg.role === 'assistant' ? (
                <div>
                  <div className="prose prose-sm max-w-none">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                  {msg.related && msg.related.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-100">
                      <p className="text-xs text-gray-500 mb-1">Related concepts:</p>
                      <div className="flex flex-wrap gap-1">
                        {msg.related.map(r => (
                          <button
                            key={r.id}
                            onClick={() => quickAsk(r.term)}
                            className="px-2 py-0.5 text-xs bg-emerald-50 text-emerald-700 rounded-full hover:bg-emerald-100"
                          >
                            {r.term}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {msg.time && (
                    <p className="text-xs text-gray-400 mt-2">{(msg.time / 1000).toFixed(1)}s</p>
                  )}
                </div>
              ) : (
                <p>{msg.content}</p>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-xl px-4 py-3 flex items-center gap-2 text-gray-500">
              <Loader2 size={16} className="animate-spin" />
              <span className="text-sm">Thinking...</span>
            </div>
          </div>
        )}
        <div ref={messagesEnd} />
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about an AI policy concept..."
          className="flex-1 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="px-4 py-3 bg-emerald-600 text-white rounded-xl hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Send size={18} />
        </button>
      </form>
    </div>
  );
}
