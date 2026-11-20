import React from 'react';

export default function PlainEnglishCard({ data }) {
  if (!data || !data.plain_english) return null;
  const p = data.plain_english;
  const isWinner = (p.verdict || '').includes('WINNER') || (p.verdict || '').includes('SIGNIFICANT') || (p.verdict || '').includes('DEPLOY_B');

  return (
    <div className={`p-4 rounded-xl border ${isWinner ? 'bg-emerald-50 border-emerald-300' : 'bg-amber-50 border-amber-300'}`}>
      <div className="flex items-center gap-2">
        <span className={`inline-block w-2.5 h-2.5 rounded-full ${isWinner ? 'bg-emerald-500' : 'bg-amber-500'}`} />
        <h3 className={`font-bold text-sm ${isWinner ? 'text-emerald-950' : 'text-amber-950'}`}>
          {p.headline || p.verdict}
        </h3>
      </div>
      <p className="text-xs text-slate-800 mt-2 leading-relaxed">
        {p.summary}
      </p>
      <div className="mt-3 pt-2.5 border-t border-slate-200/60 flex items-center justify-between text-xs font-semibold">
        <span className={isWinner ? 'text-emerald-800' : 'text-amber-800'}>
          <strong>Next Action:</strong> {p.recommendation}
        </span>
        <span className="text-slate-500 text-[11px] uppercase font-bold">{p.risk_assessment}</span>
      </div>
    </div>
  );
}
