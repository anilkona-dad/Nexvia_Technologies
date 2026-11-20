"""Main FastAPI Application for Causal Inference & A/B Testing Platform."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from backend.app.api.router import api_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("CausalPlatform")

app = FastAPI(
    title="Causal Inference & Experimentation Platform",
    description="Internal tool for A/B test design, safe sequential analysis, Bayesian evaluation, and observational causal inference.",
    version="1.0.0"
)

# CORS setup for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "Causal A/B Testing Platform", "version": "1.0.0"}


@app.get("/", response_class=HTMLResponse, tags=["Dashboard UI"])
def get_interactive_dashboard():
    """Renders the comprehensive interactive web dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Causal & A/B Experimentation Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .tab-btn.active { border-bottom: 3px solid #2563eb; color: #2563eb; font-weight: 600; }
    </style>
</head>
<body class="bg-slate-50 text-slate-900 font-sans min-h-screen">
    <!-- Navbar -->
    <header class="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="bg-blue-600 text-white p-2 rounded-lg"><i class="fa-solid fa-chart-line"></i></div>
                <div>
                    <h1 class="text-lg font-bold">Causal & A/B Testing Platform</h1>
                    <p class="text-xs text-slate-500">Internal Experimentation & Observational Causal Inference Engine</p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <a href="/docs" target="_blank" class="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-3 py-1.5 rounded font-medium border border-slate-300">
                    <i class="fa-solid fa-code"></i> Interactive Swagger API
                </a>
            </div>
        </div>
        <!-- Navigation Tabs -->
        <div class="max-w-7xl mx-auto px-6 flex space-x-8 border-t border-slate-100 overflow-x-auto">
            <button onclick="switchTab('power')" id="tab-power" class="tab-btn active py-3 text-sm flex items-center gap-2"><i class="fa-solid fa-bolt"></i> Power & Sample Size</button>
            <button onclick="switchTab('frequentist')" id="tab-frequentist" class="tab-btn py-3 text-sm flex items-center gap-2"><i class="fa-solid fa-flask"></i> Frequentist & FDR</button>
            <button onclick="switchTab('sequential')" id="tab-sequential" class="tab-btn py-3 text-sm flex items-center gap-2"><i class="fa-solid fa-clock-rotate-left"></i> Safe Sequential (mSPRT)</button>
            <button onclick="switchTab('bayesian')" id="tab-bayesian" class="tab-btn py-3 text-sm flex items-center gap-2"><i class="fa-solid fa-brain"></i> Bayesian A/B</button>
            <button onclick="switchTab('causal')" id="tab-causal" class="tab-btn py-3 text-sm flex items-center gap-2"><i class="fa-solid fa-diagram-project"></i> Observational Causal (PSM, DiD, SCM)</button>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 py-8">
        <!-- 1. POWER & SAMPLE SIZE TAB -->
        <section id="panel-power" class="space-y-6">
            <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <h2 class="text-xl font-bold text-slate-800 mb-1">Power & Sample Size Calculator (Milestones 1 & 2)</h2>
                <p class="text-sm text-slate-500 mb-6">Plan experiments with rigorous Type I (alpha) and Type II (beta) statistical error control.</p>
                
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 mb-1">Baseline Conversion Rate</label>
                        <input id="power-baseline" type="number" step="0.01" value="0.10" class="w-full border rounded-lg p-2 text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 mb-1">Relative MDE (+%)</label>
                        <input id="power-mde" type="number" step="0.01" value="0.05" class="w-full border rounded-lg p-2 text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 mb-1">Significance Level (α)</label>
                        <input id="power-alpha" type="number" step="0.01" value="0.05" class="w-full border rounded-lg p-2 text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-600 mb-1">Statistical Power (1 - β)</label>
                        <input id="power-beta" type="number" step="0.01" value="0.80" class="w-full border rounded-lg p-2 text-sm">
                    </div>
                </div>

                <button onclick="calculatePower()" class="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-lg text-sm font-semibold">
                    Compute Sample Size & Power Curve
                </button>

                <div id="power-results" class="mt-8 hidden space-y-6">
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div class="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                            <span class="text-xs font-bold text-blue-700 uppercase">Per Variant Required</span>
                            <p id="res-sample-variant" class="text-2xl font-bold text-blue-900 mt-1">-</p>
                        </div>
                        <div class="bg-indigo-50 border border-indigo-200 p-4 rounded-lg">
                            <span class="text-xs font-bold text-indigo-700 uppercase">Total Visitors Needed</span>
                            <p id="res-sample-total" class="text-2xl font-bold text-indigo-900 mt-1">-</p>
                        </div>
                        <div class="bg-emerald-50 border border-emerald-200 p-4 rounded-lg">
                            <span class="text-xs font-bold text-emerald-700 uppercase">Absolute Detectable Lift</span>
                            <p id="res-mde-abs" class="text-2xl font-bold text-emerald-900 mt-1">-</p>
                        </div>
                    </div>
                    <div class="border rounded-xl p-4 bg-white">
                        <h3 class="text-sm font-bold text-slate-700 mb-4">Statistical Power Sensitivity Curve</h3>
                        <div class="h-64"><canvas id="powerChart"></canvas></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- 2. FREQUENTIST & FDR TAB -->
        <section id="panel-frequentist" class="space-y-6 hidden">
            <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <div class="flex justify-between items-center mb-4">
                    <div>
                        <h2 class="text-xl font-bold text-slate-800">Frequentist Hypothesis Testing (Milestone 3)</h2>
                        <p class="text-sm text-slate-500">Chi-Square proportions test, Welch's t-test, and Benjamini-Hochberg FDR adjustments.</p>
                    </div>
                    <button onclick="loadCookieCatsBenchmark()" class="bg-slate-100 hover:bg-slate-200 text-slate-700 border px-3 py-1.5 rounded-lg text-xs font-semibold">
                        <i class="fa-solid fa-gamepad"></i> Load Cookie Cats A/B Benchmark
                    </button>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    <div class="border p-4 rounded-lg bg-slate-50">
                        <h3 class="font-bold text-sm text-slate-700 mb-3">Control Variant (A)</h3>
                        <div class="grid grid-cols-2 gap-3">
                            <div><label class="text-xs text-slate-600">Visitors</label><input id="freq-c-visitors" type="number" value="10000" class="w-full border rounded p-1.5 text-sm"></div>
                            <div><label class="text-xs text-slate-600">Conversions</label><input id="freq-c-conv" type="number" value="1000" class="w-full border rounded p-1.5 text-sm"></div>
                        </div>
                    </div>
                    <div class="border p-4 rounded-lg bg-slate-50">
                        <h3 class="font-bold text-sm text-slate-700 mb-3">Test Variant (B)</h3>
                        <div class="grid grid-cols-2 gap-3">
                            <div><label class="text-xs text-slate-600">Visitors</label><input id="freq-v-visitors" type="number" value="10000" class="w-full border rounded p-1.5 text-sm"></div>
                            <div><label class="text-xs text-slate-600">Conversions</label><input id="freq-v-conv" type="number" value="1090" class="w-full border rounded p-1.5 text-sm"></div>
                        </div>
                    </div>
                </div>

                <button onclick="runFrequentistChi()" class="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-lg text-sm font-semibold">
                    Execute Hypothesis Test
                </button>

                <div id="freq-results" class="mt-8 hidden space-y-4">
                    <div id="freq-plain-card" class="p-4 rounded-lg border"></div>
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">Control Rate</span><p id="res-freq-rate-c" class="text-lg font-bold mt-1">-</p></div>
                        <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">Variant Rate</span><p id="res-freq-rate-v" class="text-lg font-bold mt-1">-</p></div>
                        <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">Relative Lift</span><p id="res-freq-lift" class="text-lg font-bold mt-1">-</p></div>
                        <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">p-value</span><p id="res-freq-pval" class="text-lg font-bold mt-1">-</p></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- 3. SEQUENTIAL MONITORING TAB -->
        <section id="panel-sequential" class="space-y-6 hidden">
            <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <h2 class="text-xl font-bold text-slate-800 mb-1">Safe Sequential Testing with mSPRT (Milestone 4)</h2>
                <p class="text-sm text-slate-500 mb-6">Continuous monitoring allows peeking at results daily with mathematically guaranteed false-positive protection.</p>
                
                <div class="flex space-x-3 mb-6">
                    <button onclick="runMSPRTStream()" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-semibold">
                        Simulate Streaming Test (mSPRT)
                    </button>
                    <button onclick="runPeekingSim()" class="bg-amber-600 hover:bg-amber-700 text-white px-4 py-2 rounded-lg text-sm font-semibold">
                        <i class="fa-solid fa-triangle-exclamation"></i> Prove Peeking Problem (Simulation)
                    </button>
                </div>

                <div id="sequential-results" class="hidden space-y-6">
                    <div id="seq-plain-card" class="p-4 rounded-lg border"></div>
                    <div class="border rounded-xl p-4 bg-white">
                        <h3 class="text-sm font-bold text-slate-700 mb-2">mSPRT Likelihood Ratio vs. Stopping Boundary (1 / α = 20)</h3>
                        <div class="h-64"><canvas id="msprtChart"></canvas></div>
                    </div>
                </div>

                <div id="peeking-sim-results" class="hidden border rounded-xl p-5 bg-amber-50 border-amber-200 mt-6">
                    <h3 class="font-bold text-amber-900 mb-2 text-base">Peeking Simulation Proof (Monte Carlo, 500 experiments under H0)</h3>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm mt-3">
                        <div class="bg-white p-3 rounded border border-amber-200">
                            <span class="text-xs text-slate-500">Nominal Target Alpha</span>
                            <p class="text-xl font-bold text-slate-800">5.0%</p>
                        </div>
                        <div class="bg-white p-3 rounded border border-rose-300">
                            <span class="text-xs text-rose-600 font-bold">Naive Daily Peeking False Positive Rate</span>
                            <p id="res-peeking-naive" class="text-xl font-bold text-rose-700 mt-1">-</p>
                        </div>
                        <div class="bg-white p-3 rounded border border-emerald-300">
                            <span class="text-xs text-emerald-600 font-bold">mSPRT Protected False Positive Rate</span>
                            <p id="res-peeking-msprt" class="text-xl font-bold text-emerald-700 mt-1">-</p>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- 4. BAYESIAN A/B TAB -->
        <section id="panel-bayesian" class="space-y-6 hidden">
            <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <h2 class="text-xl font-bold text-slate-800 mb-1">Bayesian A/B Testing Engine (Milestone 5)</h2>
                <p class="text-sm text-slate-500 mb-6">Calculate the true probability that B beats A, Expected Loss, and 95% Credible Intervals.</p>
                
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                    <div><label class="text-xs text-slate-600 font-semibold">Control Visitors</label><input id="bayes-c-vis" type="number" value="12000" class="w-full border rounded p-2 text-sm"></div>
                    <div><label class="text-xs text-slate-600 font-semibold">Control Conversions</label><input id="bayes-c-conv" type="number" value="1200" class="w-full border rounded p-2 text-sm"></div>
                    <div><label class="text-xs text-slate-600 font-semibold">Variant Visitors</label><input id="bayes-v-vis" type="number" value="12000" class="w-full border rounded p-2 text-sm"></div>
                    <div><label class="text-xs text-slate-600 font-semibold">Variant Conversions</label><input id="bayes-v-conv" type="number" value="1320" class="w-full border rounded p-2 text-sm"></div>
                </div>

                <button onclick="runBayesianAB()" class="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-lg text-sm font-semibold">
                    Compute Posterior & Decisions
                </button>

                <div id="bayesian-results" class="mt-8 hidden space-y-6">
                    <div id="bayes-plain-card" class="p-4 rounded-lg border"></div>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div class="border p-4 rounded-lg bg-slate-50">
                            <span class="text-xs text-slate-500">P(Variant B > Control A)</span>
                            <p id="res-prob-b" class="text-2xl font-bold text-blue-700 mt-1">-</p>
                        </div>
                        <div class="border p-4 rounded-lg bg-slate-50">
                            <span class="text-xs text-slate-500">Expected Loss (Deploy B)</span>
                            <p id="res-exp-loss" class="text-2xl font-bold text-slate-800 mt-1">-</p>
                        </div>
                        <div class="border p-4 rounded-lg bg-slate-50">
                            <span class="text-xs text-slate-500">95% Relative Lift HDI</span>
                            <p id="res-bayes-hdi" class="text-2xl font-bold text-emerald-700 mt-1">-</p>
                        </div>
                    </div>
                    <div class="border rounded-xl p-4 bg-white">
                        <h3 class="text-sm font-bold text-slate-700 mb-2">Posterior Beta Distributions</h3>
                        <div class="h-64"><canvas id="bayesChart"></canvas></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- 5. OBSERVATIONAL CAUSAL TAB -->
        <section id="panel-causal" class="space-y-6 hidden">
            <div class="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <h2 class="text-xl font-bold text-slate-800 mb-1">Observational Causal Inference (Milestone 6)</h2>
                <p class="text-sm text-slate-500 mb-6">Estimate true cause-and-effect when experiments cannot be randomized (DoWhy, EconML, PSM, DiD, SCM).</p>
                
                <div class="flex flex-wrap gap-3 mb-6">
                    <button onclick="runCardKruegerDiD()" class="bg-slate-800 hover:bg-slate-900 text-white px-4 py-2 rounded-lg text-sm font-semibold">
                        <i class="fa-solid fa-scale-balanced"></i> Difference-in-Differences (Card & Krueger 1994)
                    </button>
                    <button onclick="runLaLondePSM()" class="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-sm font-semibold">
                        <i class="fa-solid fa-users-between-lines"></i> Propensity Score Matching (LaLonde NSW)
                    </button>
                    <button onclick="runProp99SCM()" class="bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-lg text-sm font-semibold">
                        <i class="fa-solid fa-chart-area"></i> Synthetic Control (California Prop 99)
                    </button>
                </div>

                <div id="causal-results" class="hidden space-y-6">
                    <div id="causal-plain-card" class="p-4 rounded-lg border"></div>
                    <div id="causal-metrics" class="grid grid-cols-1 md:grid-cols-4 gap-4"></div>
                    <div id="causal-chart-box" class="border rounded-xl p-4 bg-white hidden">
                        <h3 id="causal-chart-title" class="text-sm font-bold text-slate-700 mb-2"></h3>
                        <div class="h-64"><canvas id="causalChart"></canvas></div>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <script>
        let powerChartInst = null;
        let msprtChartInst = null;
        let bayesChartInst = null;
        let causalChartInst = null;

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('tab-' + tabId).classList.add('active');
            ['power', 'frequentist', 'sequential', 'bayesian', 'causal'].forEach(p => {
                document.getElementById('panel-' + p).classList.add('hidden');
            });
            document.getElementById('panel-' + tabId).classList.remove('hidden');
        }

        function renderPlainCard(cardEl, data) {
            const p = data.plain_english || {};
            const isWinner = (p.verdict || '').includes('WINNER') || (p.verdict || '').includes('SIGNIFICANT') || (p.verdict || '').includes('DEPLOY_B');
            cardEl.className = isWinner ? 'p-4 rounded-lg border bg-emerald-50 border-emerald-300' : 'p-4 rounded-lg border bg-amber-50 border-amber-300';
            cardEl.innerHTML = `
                <div class="flex items-center gap-2 font-bold ${isWinner ? 'text-emerald-900' : 'text-amber-900'}">
                    <i class="fa-solid ${isWinner ? 'fa-circle-check text-emerald-600' : 'fa-circle-info text-amber-600'}"></i>
                    <span>${p.headline || p.verdict || 'Analysis Result'}</span>
                </div>
                <p class="text-sm mt-2 text-slate-800 leading-relaxed">${p.summary || ''}</p>
                <div class="mt-3 pt-3 border-t border-slate-200/60 text-xs font-semibold ${isWinner ? 'text-emerald-800' : 'text-amber-800'}">
                    <strong>Recommendation:</strong> ${p.recommendation || ''}
                </div>
            `;
        }

        async function calculatePower() {
            const baseline = parseFloat(document.getElementById('power-baseline').value);
            const mde = parseFloat(document.getElementById('power-mde').value);
            const alpha = parseFloat(document.getElementById('power-alpha').value);
            const beta = parseFloat(document.getElementById('power-beta').value);

            const res = await fetch('/api/v1/power/sample-size-proportion', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({baseline_rate: baseline, mde_relative: mde, alpha: alpha, power: beta})
            });
            const data = await res.json();

            document.getElementById('res-sample-variant').innerText = data.sample_size_control.toLocaleString();
            document.getElementById('res-sample-total').innerText = data.total_sample_size.toLocaleString();
            document.getElementById('res-mde-abs').innerText = (data.mde_absolute * 100).toFixed(2) + '%';
            document.getElementById('power-results').classList.remove('hidden');

            const curveRes = await fetch('/api/v1/power/power-curve', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({baseline_rate: baseline, mde_relative: mde, alpha: alpha})
            });
            const curveData = await curveRes.json();

            if (powerChartInst) powerChartInst.destroy();
            const ctx = document.getElementById('powerChart').getContext('2d');
            powerChartInst = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: curveData.map(d => d.sample_size_per_variant),
                    datasets: [{
                        label: 'Statistical Power (1 - β)',
                        data: curveData.map(d => d.statistical_power),
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37,99,235,0.1)',
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 1 } } }
            });
        }

        async function runFrequentistChi() {
            const body = {
                visitors_control: parseInt(document.getElementById('freq-c-visitors').value),
                conversions_control: parseInt(document.getElementById('freq-c-conv').value),
                visitors_variant: parseInt(document.getElementById('freq-v-visitors').value),
                conversions_variant: parseInt(document.getElementById('freq-v-conv').value),
                alpha: 0.05
            };
            const res = await fetch('/api/v1/frequentist/chi-square', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)
            });
            const data = await res.json();

            document.getElementById('res-freq-rate-c').innerText = (data.control.conversion_rate * 100).toFixed(2) + '%';
            document.getElementById('res-freq-rate-v').innerText = (data.variant.conversion_rate * 100).toFixed(2) + '%';
            document.getElementById('res-freq-lift').innerText = (data.difference.relative_lift_pct > 0 ? '+' : '') + data.difference.relative_lift_pct + '%';
            document.getElementById('res-freq-pval').innerText = data.p_value.toFixed(4);

            renderPlainCard(document.getElementById('freq-plain-card'), data);
            document.getElementById('freq-results').classList.remove('hidden');
        }

        async function loadCookieCatsBenchmark() {
            const res = await fetch('/api/v1/frequentist/benchmark-cookie-cats');
            const data = await res.json();
            const test = data.retention_7_day_test;
            document.getElementById('freq-c-visitors').value = test.control.visitors;
            document.getElementById('freq-c-conv').value = test.control.conversions;
            document.getElementById('freq-v-visitors').value = test.variant.visitors;
            document.getElementById('freq-v-conv').value = test.variant.conversions;
            runFrequentistChi();
        }

        async function runMSPRTStream() {
            const N = 800;
            const a = Array.from({length: N}, () => Math.random() < 0.10 ? 1 : 0);
            const b = Array.from({length: N}, () => Math.random() < 0.135 ? 1 : 0);

            const res = await fetch('/api/v1/sequential/msprt', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({conversions_a: a, conversions_b: b, alpha: 0.05})
            });
            const data = await res.json();

            renderPlainCard(document.getElementById('seq-plain-card'), data);
            document.getElementById('sequential-results').classList.remove('hidden');

            if (msprtChartInst) msprtChartInst.destroy();
            const ctx = document.getElementById('msprtChart').getContext('2d');
            msprtChartInst = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.trajectory.map(d => d.step),
                    datasets: [
                        {
                            label: 'Likelihood Ratio (mSPRT)',
                            data: data.trajectory.map(d => d.likelihood_ratio),
                            borderColor: '#2563eb',
                            tension: 0.2
                        },
                        {
                            label: 'Stopping Boundary (1 / α = 20)',
                            data: data.trajectory.map(() => 20.0),
                            borderColor: '#e11d48',
                            borderDash: [5, 5],
                            pointRadius: 0
                        }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
        }

        async function runPeekingSim() {
            const res = await fetch('/api/v1/sequential/simulate-peeking-problem', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({sample_size: 1000, peeks: 15, num_simulations: 300})
            });
            const data = await res.json();
            document.getElementById('res-peeking-naive').innerText = (data.empirical_false_positive_naive * 100).toFixed(1) + '% (vs 5% target)';
            document.getElementById('res-peeking-msprt').innerText = (data.empirical_false_positive_msprt * 100).toFixed(1) + '% (well protected)';
            document.getElementById('peeking-sim-results').classList.remove('hidden');
        }

        async function runBayesianAB() {
            const body = {
                visitors_a: parseInt(document.getElementById('bayes-c-vis').value),
                conversions_a: parseInt(document.getElementById('bayes-c-conv').value),
                visitors_b: parseInt(document.getElementById('bayes-v-vis').value),
                conversions_b: parseInt(document.getElementById('bayes-v-conv').value)
            };
            const res = await fetch('/api/v1/bayesian/beta-binomial', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)
            });
            const data = await res.json();

            document.getElementById('res-prob-b').innerText = (data.probability_b_beats_a * 100).toFixed(1) + '%';
            document.getElementById('res-exp-loss').innerText = (data.expected_loss_deploy_b * 100).toFixed(4) + '%';
            document.getElementById('res-bayes-hdi').innerText = `[${data.relative_lift_pct_95_hdi[0]}%, ${data.relative_lift_pct_95_hdi[1]}%]`;

            renderPlainCard(document.getElementById('bayes-plain-card'), data);
            document.getElementById('bayesian-results').classList.remove('hidden');

            if (bayesChartInst) bayesChartInst.destroy();
            const ctx = document.getElementById('bayesChart').getContext('2d');
            bayesChartInst = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.pdf_curve.x.map(v => (v * 100).toFixed(2) + '%'),
                    datasets: [
                        { label: 'Control (Beta Posterior)', data: data.pdf_curve.pdf_control, borderColor: '#64748b' },
                        { label: 'Variant (Beta Posterior)', data: data.pdf_curve.pdf_variant, borderColor: '#2563eb' }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
        }

        async function runCardKruegerDiD() {
            const res = await fetch('/api/v1/causal/benchmark-card-krueger');
            const data = await res.json();
            renderPlainCard(document.getElementById('causal-plain-card'), data);

            const m = document.getElementById('causal-metrics');
            m.innerHTML = `
                <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">DiD Effect</span><p class="text-lg font-bold text-slate-900 mt-1">${data.did_treatment_effect}</p></div>
                <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">Standard Error</span><p class="text-lg font-bold text-slate-900 mt-1">${data.standard_error}</p></div>
                <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">t-statistic</span><p class="text-lg font-bold text-slate-900 mt-1">${data.t_statistic}</p></div>
                <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">p-value</span><p class="text-lg font-bold text-slate-900 mt-1">${data.p_value}</p></div>
            `;
            document.getElementById('causal-results').classList.remove('hidden');

            document.getElementById('causal-chart-box').classList.remove('hidden');
            document.getElementById('causal-chart-title').innerText = 'Difference-in-Differences Parallel Trends Trajectory';
            if (causalChartInst) causalChartInst.destroy();
            const ctx = document.getElementById('causalChart').getContext('2d');
            causalChartInst = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Pre-Period (Feb 1992)', 'Post-Period (Nov 1992)'],
                    datasets: [
                        { label: 'NJ (Treated State)', data: [data.group_means.treated_pre, data.group_means.treated_post], borderColor: '#2563eb' },
                        { label: 'PA (Control State)', data: [data.group_means.control_pre, data.group_means.control_post], borderColor: '#64748b' },
                        { label: 'NJ Counterfactual', data: [data.group_means.treated_pre, data.group_means.counterfactual_treated_post], borderColor: '#9333ea', borderDash: [4, 4] }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
        }

        async function runLaLondePSM() {
            const res = await fetch('/api/v1/causal/benchmark-lalonde');
            const data = await res.json();
            renderPlainCard(document.getElementById('causal-plain-card'), data);

            const m = document.getElementById('causal-metrics');
            m.innerHTML = `
                <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">Matched ATT</span><p class="text-lg font-bold text-slate-900 mt-1">$${data.att.toLocaleString()}</p></div>
                <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">Matched Pairs</span><p class="text-lg font-bold text-slate-900 mt-1">${data.matched_pairs}</p></div>
                <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">p-value</span><p class="text-lg font-bold text-slate-900 mt-1">${data.p_value}</p></div>
                <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">DoWhy Refutation</span><p class="text-lg font-bold text-emerald-700 mt-1">${data.dowhy_refutation ? 'PASSED' : 'UNVERIFIED'}</p></div>
            `;
            document.getElementById('causal-results').classList.remove('hidden');
            document.getElementById('causal-chart-box').classList.add('hidden');
        }

        async function runProp99SCM() {
            const res = await fetch('/api/v1/causal/benchmark-prop99');
            const data = await res.json();
            renderPlainCard(document.getElementById('causal-plain-card'), data);

            const m = document.getElementById('causal-metrics');
            m.innerHTML = `
                <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">Post-Prop 99 Effect</span><p class="text-lg font-bold text-slate-900 mt-1">${data.average_post_treatment_effect} packs</p></div>
                <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">Pre-Period RMSE</span><p class="text-lg font-bold text-slate-900 mt-1">${data.pre_intervention_rmse}</p></div>
                <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">Treated Unit</span><p class="text-lg font-bold text-slate-900 mt-1">${data.treated_unit}</p></div>
                <div class="border p-3 rounded-lg"><span class="text-xs text-slate-500">Intervention Year</span><p class="text-lg font-bold text-slate-900 mt-1">${data.intervention_time}</p></div>
            `;
            document.getElementById('causal-results').classList.remove('hidden');

            document.getElementById('causal-chart-box').classList.remove('hidden');
            document.getElementById('causal-chart-title').innerText = 'California Cigarette Sales vs. Synthetic California (1970 - 2000)';
            if (causalChartInst) causalChartInst.destroy();
            const ctx = document.getElementById('causalChart').getContext('2d');
            causalChartInst = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.timeline.map(d => d.time),
                    datasets: [
                        { label: 'California (Actual)', data: data.timeline.map(d => d.actual), borderColor: '#2563eb' },
                        { label: 'Synthetic California (Convex Donor Pool)', data: data.timeline.map(d => d.synthetic), borderColor: '#e11d48', borderDash: [4, 4] }
                    ]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
