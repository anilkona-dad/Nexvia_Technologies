import React, { useState } from 'react';
import PlainEnglishCard from './components/PlainEnglishCard';

export default function App() {
  const [activeTab, setActiveTab] = useState('power');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Form states
  const [baseline, setBaseline] = useState(0.10);
  const [mde, setMde] = useState(0.05);

  const calculatePower = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/v1/power/sample-size-proportion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ baseline_rate: parseFloat(baseline), mde_relative: parseFloat(mde) })
      });
      const data = await res.json();
      setResult(data);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-600 text-white p-2 rounded-lg font-bold">CA</div>
            <div>
              <h1 className="text-base font-bold text-slate-800">Causal & A/B Experimentation Platform</h1>
              <p className="text-xs text-slate-500">FastAPI & React Internal Decision Suite</p>
            </div>
          </div>
        </div>
        <div className="max-w-6xl mx-auto px-6 flex space-x-6 border-t border-slate-100 text-sm">
          {['power', 'frequentist', 'sequential', 'bayesian', 'causal'].map((t) => (
            <button
              key={t}
              onClick={() => setActiveTab(t)}
              className={`py-3 capitalize font-medium border-b-2 ${
                activeTab === t ? 'border-blue-600 text-blue-600' : 'border-transparent text-slate-500'
              }`}
            >
              {t === 'power' ? 'Power Calculator' : t === 'frequentist' ? 'Frequentist & FDR' : t === 'sequential' ? 'Sequential mSPRT' : t === 'bayesian' ? 'Bayesian A/B' : 'Observational Causal'}
            </button>
          ))}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm space-y-6">
          <h2 className="text-lg font-bold text-slate-800 capitalize">
            {activeTab} Analytics Module
          </h2>

          {activeTab === 'power' && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-slate-600">Baseline Conversion Rate</label>
                  <input
                    type="number"
                    step="0.01"
                    value={baseline}
                    onChange={(e) => setBaseline(e.target.value)}
                    className="w-full border rounded-lg p-2 text-sm mt-1"
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-600">Relative MDE</label>
                  <input
                    type="number"
                    step="0.01"
                    value={mde}
                    onChange={(e) => setMde(e.target.value)}
                    className="w-full border rounded-lg p-2 text-sm mt-1"
                  />
                </div>
              </div>
              <button
                onClick={calculatePower}
                disabled={loading}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-semibold hover:bg-blue-700"
              >
                {loading ? 'Computing...' : 'Calculate Sample Size'}
              </button>

              {result && (
                <div className="mt-4 p-4 bg-slate-50 rounded-lg border grid grid-cols-3 gap-4">
                  <div>
                    <span className="text-xs text-slate-500">Per Variant Size</span>
                    <p className="text-xl font-bold text-blue-600">{result.sample_size_control?.toLocaleString()}</p>
                  </div>
                  <div>
                    <span className="text-xs text-slate-500">Total Sample Needed</span>
                    <p className="text-xl font-bold text-slate-800">{result.total_sample_size?.toLocaleString()}</p>
                  </div>
                  <div>
                    <span className="text-xs text-slate-500">Absolute MDE</span>
                    <p className="text-xl font-bold text-emerald-600">{(result.mde_absolute * 100)?.toFixed(2)}%</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab !== 'power' && (
            <div className="text-sm text-slate-600 space-y-2">
              <p>For full interactive visualizations, sequential mSPRT charts, and observational causal methods (PSM, DiD, SCM), open the integrated dashboard served directly by the backend at <strong>http://localhost:8000/</strong>.</p>
            </div>
          )}

          {result && <PlainEnglishCard data={result} />}
        </div>
      </main>
    </div>
  );
}
