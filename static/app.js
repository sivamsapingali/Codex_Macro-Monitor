// State
let dashboardData = {};
let intelligenceData = {};
let predictionData = {};
let usageData = {};
let backtestData = {};
let currentCategory = 'growth';
let railChartInstance = null;
let methodChartInstance = null;
let backtestChartInstance = null;

// --- Initialization ---

document.addEventListener('DOMContentLoaded', async () => {
    setupSidebar();
    startClock();

    await loadStatus();
    await Promise.all([
        loadIntelligence(),
        loadDashboard(),
        loadPredictions(),
        loadUsage()
    ]);
    await loadBacktest();

    setInterval(async () => {
        await Promise.all([
            loadIntelligence(),
            loadDashboard(),
            loadPredictions(),
            loadUsage()
        ]);
    }, 60000);

    setInterval(loadBacktest, 10 * 60 * 1000);
});

async function forceUpdate() {
    try {
        await Promise.all([
            fetch('/api/admin/refresh-stale'),
            fetch('/api/admin/refresh-stale-market')
        ]);
        alert('Sync triggered (stale-only). Data will refresh shortly.');
    } catch (e) { console.error(e); }
}

// --- Navigation ---

function setupSidebar() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            if (item.dataset.view) {
                switchView(item.dataset.view);
            }
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
        });
    });
}

function switchView(viewName) {
    document.querySelectorAll('.content-view').forEach(el => el.style.display = 'none');
    const target = document.getElementById(`view-${viewName}`);
    if (target) target.style.display = 'block';

    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const navItem = document.querySelector(`.nav-item[data-view="${viewName}"]`);
    if (navItem) navItem.classList.add('active');
}

function switchCategory(catId) {
    switchView('dashboard');
    currentCategory = catId;
    renderCategory(catId);

    const map = {
        'growth': 'Growth & Output',
        'inflation': 'Inflation & Prices',
        'labor': 'Labor Market',
        'rates': 'Yields & Bonds',
        'money': 'Liquidity & Credit',
        'housing': 'Housing & Real Estate',
        'commodities': 'Commodities',
        'markets': 'Markets & Risk'
    };
    const title = document.getElementById('category-title');
    if (title) title.textContent = map[catId] || 'Data Series';
}

// --- Data Loading ---

async function loadStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const statusText = document.getElementById('status-text');
        const statusMeta = document.getElementById('status-meta');
        if (statusText) statusText.textContent = data.status === 'online' ? 'System Online' : 'System Offline';
        if (statusMeta) {
            const mode = data.data_mode ? ` • ${data.data_mode}` : '';
            const provider = data.market_provider ? ` • ${data.market_provider}` : '';
            statusMeta.textContent = `${data.data_points || '--'} series${mode}${provider}`;
        }
    } catch (e) { console.error(e); }
}

async function loadIntelligence() {
    try {
        const res = await fetch('/api/intelligence');
        intelligenceData = await res.json();
        renderIntelligence();
        renderSignalsSidebar();
        renderSectors();
    } catch (e) { console.error(e); }
}

async function loadPredictions() {
    try {
        const res = await fetch('/api/predictions');
        predictionData = await res.json();
        renderSectors();
        renderActionableItems();
        renderTradeIdeas();
        renderMethodology();
    } catch (e) { console.error(e); }
}

async function loadBacktest() {
    try {
        const res = await fetch('/api/backtest');
        backtestData = await res.json();
        renderBacktest();
    } catch (e) { console.error(e); }
}

async function loadUsage() {
    try {
        const res = await fetch('/api/usage');
        usageData = await res.json();
        renderMethodology();
    } catch (e) { console.error(e); }
}

async function loadDashboard() {
    try {
        const res = await fetch('/api/dashboard');
        dashboardData = await res.json();
        renderCategory(currentCategory);
        renderKeyMetrics();
        renderSideMetrics();
        loadRailChart();
    } catch (e) { console.error(e); }
}

// --- Rendering ---

function renderIntelligence() {
    const regime = intelligenceData.regime || {};
    const chip = document.getElementById('regime-chip');
    const regimeMain = document.getElementById('regime-main');

    if (chip) chip.textContent = regime.name || 'Unknown';
    if (regimeMain) {
        regimeMain.innerHTML = `
            <div class="regime-title">${(regime.name || 'Unknown').toUpperCase()}</div>
            <div class="regime-desc">${regime.description || 'Insufficient data to determine regime.'}</div>
            <div class="regime-trends">
                <span class="${regime.growth_trend === 'Rising' ? 'text-green' : 'text-red'}">Growth ${regime.growth_trend || 'Unknown'}</span>
                <span class="${regime.inflation_trend === 'Rising' ? 'text-red' : 'text-green'}">Inflation ${regime.inflation_trend || 'Unknown'}</span>
            </div>
        `;
    }

    const anomEl = document.getElementById('anomalies-list');
    if (anomEl) {
        const anoms = intelligenceData.anomalies || [];
        if (!anoms.length) {
            anomEl.textContent = 'No anomalies detected';
        } else {
            anomEl.innerHTML = '';
            anoms.slice(0, 4).forEach(a => {
                const div = document.createElement('div');
                div.className = 'anomaly-item';
                div.innerHTML = `
                    <span>${a.name}</span>
                    <span class="${Math.abs(a.z_score) > 3 ? 'text-red' : 'text-yellow'}">${a.z_score > 0 ? '+' : ''}${a.z_score}σ</span>
                `;
                anomEl.appendChild(div);
            });
        }
    }
}

function renderSignalsSidebar() {
    const sigEl = document.getElementById('signals-sidebar');
    if (!sigEl) return;

    const signals = intelligenceData.signals || [];
    if (!signals.length) {
        sigEl.textContent = 'No active signals';
        return;
    }
    sigEl.innerHTML = '';
    signals.forEach(sig => {
        const div = document.createElement('div');
        div.className = 'signal-item';
        div.innerHTML = `${sig.name}: ${sig.desc}`;
        sigEl.appendChild(div);
    });
}

function renderKeyMetrics() {
    const container = document.getElementById('key-metrics');
    if (!container) return;

    const specs = [
        { id: 'DGS10', label: '10Y Yield', category: 'rates', unit: '%' },
        { id: 'CPIAUCSL', label: 'CPI YoY', category: 'inflation', unit: '%', useTransformed: true },
        { id: 'UNRATE', label: 'Unemp', category: 'labor', unit: '%' },
        { id: 'DCOILWTICO', label: 'WTI', category: 'commodities', unit: '$' },
        { id: 'FEDFUNDS', label: 'Fed Funds', category: 'rates', unit: '%' },
        { id: 'T10Y2Y', label: 'Curve', category: 'rates', unit: '%' }
    ];

    container.innerHTML = '';
    specs.forEach(spec => {
        const item = getSeriesItem(spec.category, spec.id);
        const value = spec.useTransformed ? item?.value_transformed : item?.value;
        const change = item?.change;
        const changeClass = change > 0 ? 'text-green' : change < 0 ? 'text-red' : 'text-muted';
        const formatted = formatValue(value, spec.unit);
        const delta = formatDelta(change, spec.unit);

        const div = document.createElement('div');
        div.className = 'metric-item';
        div.innerHTML = `
            <div class="metric-label">${spec.label}</div>
            <div class="metric-value">${formatted}</div>
            <div class="metric-change ${changeClass}">${delta}</div>
        `;
        container.appendChild(div);
    });
}

function renderSideMetrics() {
    const container = document.getElementById('side-metrics');
    if (!container) return;

    const specs = [
        { id: 'DGS10', label: '10Y Yield', category: 'rates', unit: '%' },
        { id: 'FEDFUNDS', label: 'Fed Funds', category: 'rates', unit: '%' },
        { id: 'DCOILWTICO', label: 'WTI Crude', category: 'commodities', unit: '$' },
        { id: 'UNRATE', label: 'Unemployment', category: 'labor', unit: '%' },
        { id: 'CPIAUCSL', label: 'CPI YoY', category: 'inflation', unit: '%', useTransformed: true },
        { id: 'T10Y2Y', label: 'Yield Curve', category: 'rates', unit: '%' }
    ];

    container.innerHTML = '';
    specs.forEach(spec => {
        const item = getSeriesItem(spec.category, spec.id);
        const value = spec.useTransformed ? item?.value_transformed : item?.value;
        const change = item?.change;
        const changeClass = change > 0 ? 'text-green' : change < 0 ? 'text-red' : 'text-muted';

        const div = document.createElement('div');
        div.className = 'rail-metric-card';
        div.innerHTML = `
            <div class="rail-title">${spec.label}</div>
            <div class="rail-value">${formatValue(value, spec.unit)}</div>
            <div class="rail-change ${changeClass}">${formatDelta(change, spec.unit)}</div>
        `;
        container.appendChild(div);
    });
}

function renderCategory(catId) {
    const container = document.getElementById('content-panels');
    if (!container || !dashboardData[catId]) return;

    container.innerHTML = '';
    Object.entries(dashboardData[catId].data).forEach(([seriesId, item]) => {
        container.appendChild(createIndicatorCard(seriesId, item));
    });
}

function renderSectors() {
    const grid = document.getElementById('sector-grid');
    if (!grid) return;

    if (predictionData && predictionData.predictions && !predictionData.error) {
        const predictions = Object.values(predictionData.predictions);
        if (!predictions.length) {
            grid.innerHTML = '<div class="loading">No prediction data available.</div>';
            return;
        }

        const horizonOrder = getHorizonOrder(predictionData.horizons);
        grid.innerHTML = '';
        predictions.forEach(sector => {
            const card = document.createElement('div');
            card.className = 'sector-card';

            const horizonRows = horizonOrder
                .filter(h => sector.horizons && sector.horizons[h])
                .map(h => {
                    const item = sector.horizons[h];
                    const dirUp = item.direction === 'UP';
                    const dirSymbol = dirUp ? 'UP' : 'DOWN';
                    const dirClass = dirUp ? 'text-green' : 'text-red';
                    const conf = Math.round((item.confidence || 0) * 100);
                    return `
                        <div class="horizon-row">
                            <span class="horizon-label">${h}</span>
                            <span class="horizon-dir ${dirClass}">${dirSymbol}</span>
                            <span class="horizon-conf">${conf}%</span>
                        </div>
                    `;
                }).join('');

            const topDrivers = (sector.horizons && sector.horizons['1m'] ? sector.horizons['1m'].top_drivers : [])
                .slice(0, 2)
                .map(d => d.series)
                .join(' • ');

            const lastPrice = typeof sector.last_price === 'number' ? sector.last_price.toFixed(2) : '--';
            card.innerHTML = `
                <div class="sector-icon">${sector.icon || 'S'}</div>
                <div class="sector-info">
                    <div class="sector-name">${sector.name}<span class="sector-ticker">${sector.symbol}</span></div>
                    <div class="text-muted">Last: ${lastPrice} (${sector.last_date || '--'})</div>
                    <div class="text-muted">${topDrivers || 'Top drivers: acceleration + deceleration mix'}</div>
                </div>
                <div class="sector-predictions">
                    ${horizonRows}
                </div>
            `;

            grid.appendChild(card);
        });
        return;
    }

    grid.innerHTML = '<div class="loading">Prediction engine warming up...</div>';
}

function renderActionableItems() {
    const container = document.getElementById('actionable-items');
    if (!container) return;

    if (!predictionData || !predictionData.ranked) {
        container.innerHTML = '<div class="loading">Building action list...</div>';
        return;
    }

    const horizons = getHorizonOrder();
    container.innerHTML = '';

    horizons.forEach(horizon => {
        const ranked = predictionData.ranked[horizon];
        if (!ranked) return;

        const bullish = ranked.long || ranked.bullish || [];
        const bearish = ranked.short || ranked.bearish || [];
        const watchlist = Object.values(predictionData.predictions || {})
            .map(sector => {
                const h = sector.horizons?.[horizon];
                return h ? { symbol: sector.symbol, confidence: h.confidence, direction: h.direction } : null;
            })
            .filter(Boolean)
            .filter(item => item.confidence >= 0.45 && item.confidence <= 0.55)
            .slice(0, 3);

        const card = document.createElement('div');
        card.className = 'action-card';
        card.innerHTML = `
            <div class="action-title">Horizon ${horizon}</div>
            <div>
                <div class="action-title">Overweight</div>
                <div class="action-list">
                    ${bullish.map(item => buildActionPill(item.symbol, item.confidence, 'UP')).join('') || '<span class="text-muted">None</span>'}
                </div>
            </div>
            <div>
                <div class="action-title">Underweight</div>
                <div class="action-list">
                    ${bearish.map(item => buildActionPill(item.symbol, item.confidence, 'DOWN')).join('') || '<span class="text-muted">None</span>'}
                </div>
            </div>
            <div>
                <div class="action-title">Watchlist</div>
                <div class="action-list">
                    ${watchlist.map(item => buildActionPill(item.symbol, item.confidence, item.direction)).join('') || '<span class="text-muted">None</span>'}
                </div>
            </div>
        `;
        container.appendChild(card);
    });
}

function renderTradeIdeas() {
    const container = document.getElementById('trade-ideas');
    if (!container) return;

    if (!predictionData || !predictionData.ranked) {
        container.innerHTML = '<div class="loading">Ranking trade ideas...</div>';
        return;
    }

    const horizons = getHorizonOrder();
    container.innerHTML = '';

    horizons.forEach(horizon => {
        const ranked = predictionData.ranked[horizon];
        if (!ranked) return;
        const bullish = ranked.long || ranked.bullish || [];
        const bearish = ranked.short || ranked.bearish || [];
        const items = [
            ...bullish.map(item => ({ ...item, direction: 'UP' })),
            ...bearish.map(item => ({ ...item, direction: 'DOWN' })),
        ];
        if (!items.length) return;

        const maxAbs = Math.max(
            ...items.map(item => Math.abs(item.expected_pnl || 0)),
            1
        );

        const card = document.createElement('div');
        card.className = 'trade-card';
        card.innerHTML = `
            <div class="action-title">Horizon ${horizon}</div>
            ${items.map(item => buildTradeRow(item, maxAbs)).join('')}
        `;
        container.appendChild(card);
    });
}

function buildTradeRow(item, maxAbs) {
    const expected = item.expected_return || 0;
    const expectedPnl = item.expected_pnl || 0;
    const weight = item.position_weight || 0;
    const direction = item.direction || (expected >= 0 ? 'UP' : 'DOWN');
    const dirClass = direction === 'DOWN' ? 'down' : '';
    const width = Math.min(100, Math.abs(expectedPnl) / maxAbs * 100);
    const conf = Math.round((item.confidence || 0) * 100);
    return `
        <div class="trade-row">
            <div class="trade-symbol">
                <span>${item.symbol}</span>
                <span class="trade-direction">${direction === 'DOWN' ? 'Short' : 'Long'}</span>
            </div>
            <div class="trade-bar">
                <div class="trade-bar-fill ${dirClass}" style="width: ${width}%;"></div>
            </div>
            <div class="trade-metrics">
                <span>${formatCurrency(expectedPnl)}</span>
                <span>${formatPercent(weight)} size</span>
                <span>${conf}% conf</span>
            </div>
        </div>
    `;
}

function renderMethodology() {
    const usageEl = document.getElementById('data-usage');
    const powerEl = document.getElementById('model-power');
    const summaryEl = document.getElementById('model-summary');

    if (usageEl) {
        const entries = Object.values(usageData || {});
        if (!entries.length) {
            usageEl.innerHTML = '<div class="loading">Usage data loading...</div>';
        } else {
            usageEl.innerHTML = '';
            entries.forEach(entry => {
                const limit = entry.daily_limit === null ? 'n/a' : entry.daily_limit;
                const remaining = entry.remaining === null ? 'n/a' : entry.remaining;
                const div = document.createElement('div');
                div.className = 'usage-item';
                div.innerHTML = `
                    <span>${entry.provider}</span>
                    <span>${entry.count}/${limit} (rem ${remaining})</span>
                `;
                usageEl.appendChild(div);
            });
        }
    }

    if (powerEl) {
        const metrics = collectModelMetrics();
        if (!metrics) {
            powerEl.innerHTML = '<div class="loading">Model metrics loading...</div>';
        } else {
            const rows = buildPowerRows(metrics);
            const asOf = predictionData?.as_of ? `As of ${predictionData.as_of}` : '';
            const mix = metrics.model_mix ? `Model mix: ${metrics.model_mix}` : '';
            powerEl.innerHTML = `${rows}<div class="text-muted">${[asOf, mix].filter(Boolean).join(' | ')}</div>`;
        }
    }

    if (summaryEl) {
        const stats = computeHorizonStats();
        if (!stats) {
            summaryEl.textContent = 'Model metrics loading...';
        } else {
            const avgBal = average(Object.values(stats).map(s => s.balanced_accuracy || s.accuracy));
            const avgAuc = average(Object.values(stats).map(s => s.auc || 0));
            const avgEdge = avgBal - 0.5;
            summaryEl.textContent = `Avg balanced accuracy: ${(avgBal * 100).toFixed(1)}% (edge ${(avgEdge * 100).toFixed(1)}%), avg AUC ${(avgAuc * 100).toFixed(1)}%.`;
            renderMethodologyChart(stats);
        }
    }
}

function renderBacktest() {
    const summaryEl = document.getElementById('backtest-summary');
    const chartEl = document.getElementById('backtest-chart');
    if (!summaryEl || !chartEl) return;

    if (!backtestData || !backtestData.backtests) {
        summaryEl.textContent = 'Backtest loading...';
        return;
    }

    const backtests = backtestData.backtests || {};
    const window = backtestData.window || {};
    const horizonOrder = getHorizonOrder();
    let horizons = horizonOrder.filter(h => backtests[h] && backtests[h].equity_curve?.length);
    if (!horizons.length) {
        horizons = Object.keys(backtests).filter(h => backtests[h].equity_curve?.length).slice(0, 3);
    }

    const summaries = Object.entries(backtests)
        .map(([h, data]) => ({ horizon: h, stats: data.stats || {} }))
        .filter(item => item.stats.total_pnl !== undefined);

    if (!summaries.length) {
        summaryEl.textContent = 'Backtest unavailable for current data window.';
    } else {
        summaries.sort((a, b) => (b.stats.total_return || 0) - (a.stats.total_return || 0));
        const best = summaries[0];
        const windowText = window.start && window.end ? `Window ${window.start} → ${window.end}` : '';
        summaryEl.textContent = `Best horizon ${best.horizon}: ${formatCurrency(best.stats.total_pnl || 0)} total P&L, drawdown ${formatCurrency(best.stats.max_drawdown || 0)}, hit rate ${formatPercent(best.stats.hit_rate || 0)} (${best.stats.trades || 0} trades). ${windowText}`;
    }

    const dateSet = new Set();
    horizons.forEach(h => {
        (backtests[h].equity_curve || []).forEach(pt => dateSet.add(pt.date));
    });
    const labels = Array.from(dateSet).sort();

    const colors = ['rgba(44, 255, 159, 0.8)', 'rgba(120, 200, 255, 0.8)', 'rgba(242, 95, 92, 0.8)'];
    const datasets = horizons.map((h, idx) => {
        const curveMap = new Map((backtests[h].equity_curve || []).map(pt => [pt.date, pt.equity]));
        const data = labels.map(label => curveMap.get(label) ?? null);
        return {
            label: h,
            data,
            borderColor: colors[idx % colors.length],
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.25,
            spanGaps: true,
        };
    });

    if (backtestChartInstance) {
        backtestChartInstance.destroy();
    }

    backtestChartInstance = new Chart(chartEl, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: true, labels: { color: '#7a8597' } } },
            scales: {
                x: { ticks: { color: '#7a8597' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { ticks: { color: '#7a8597' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            }
        }
    });
}

function collectModelMetrics() {
    if (!predictionData || !predictionData.predictions) return null;

    const horizons = Object.keys(predictionData.horizons || {});
    let acc = 0;
    let balanced = 0;
    let precision = 0;
    let recall = 0;
    let auc = 0;
    let aucCount = 0;
    let count = 0;
    const modelCounts = {};

    Object.values(predictionData.predictions).forEach(sector => {
        horizons.forEach(h => {
            const metrics = sector.horizons?.[h]?.metrics;
            if (!metrics) return;
            acc += metrics.accuracy || 0;
            balanced += metrics.balanced_accuracy || 0;
            precision += metrics.precision || 0;
            recall += metrics.recall || 0;
            if (metrics.auc !== null && metrics.auc !== undefined) {
                auc += metrics.auc || 0;
                aucCount += 1;
            }
            if (metrics.model) {
                modelCounts[metrics.model] = (modelCounts[metrics.model] || 0) + 1;
            }
            count += 1;
        });
    });

    if (!count) return null;
    const modelMix = Object.entries(modelCounts)
        .sort((a, b) => b[1] - a[1])
        .map(([model, n]) => `${model}:${n}`)
        .join(' • ');
    return {
        accuracy: acc / count,
        balanced_accuracy: balanced / count,
        precision: precision / count,
        recall: recall / count,
        auc: aucCount ? auc / aucCount : null,
        model_mix: modelMix || null,
        samples: count
    };
}

function computeHorizonStats() {
    if (!predictionData || !predictionData.predictions || !predictionData.horizons) return null;
    const horizons = Object.keys(predictionData.horizons);
    const stats = {};

    horizons.forEach(h => {
        stats[h] = { accuracy: 0, balanced_accuracy: 0, precision: 0, recall: 0, auc: 0, aucCount: 0, count: 0 };
    });

    Object.values(predictionData.predictions).forEach(sector => {
        horizons.forEach(h => {
            const metrics = sector.horizons?.[h]?.metrics;
            if (!metrics) return;
            stats[h].accuracy += metrics.accuracy || 0;
            stats[h].balanced_accuracy += metrics.balanced_accuracy || 0;
            stats[h].precision += metrics.precision || 0;
            stats[h].recall += metrics.recall || 0;
            if (metrics.auc !== null && metrics.auc !== undefined) {
                stats[h].auc += metrics.auc || 0;
                stats[h].aucCount += 1;
            }
            stats[h].count += 1;
        });
    });

    horizons.forEach(h => {
        if (stats[h].count) {
            stats[h].accuracy /= stats[h].count;
            stats[h].balanced_accuracy /= stats[h].count;
            stats[h].precision /= stats[h].count;
            stats[h].recall /= stats[h].count;
            stats[h].auc = stats[h].aucCount ? stats[h].auc / stats[h].aucCount : null;
        }
    });

    return stats;
}

function renderMethodologyChart(stats) {
    const ctx = document.getElementById('method-accuracy-chart');
    if (!ctx) return;
    const labels = Object.keys(stats);
    const accuracy = labels.map(label => stats[label].balanced_accuracy || stats[label].accuracy || 0);
    const auc = labels.map(label => stats[label].auc || 0);
    const baseline = labels.map(() => 0.5);

    if (methodChartInstance) {
        methodChartInstance.destroy();
    }

    methodChartInstance = new Chart(ctx, {
        data: {
            labels,
            datasets: [
                {
                    type: 'bar',
                    label: 'Balanced Accuracy',
                    data: accuracy,
                    backgroundColor: 'rgba(44, 255, 159, 0.4)',
                    borderColor: 'rgba(44, 255, 159, 0.8)',
                    borderWidth: 1,
                },
                {
                    type: 'line',
                    label: 'AUC',
                    data: auc,
                    borderColor: 'rgba(120, 200, 255, 0.7)',
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    type: 'line',
                    label: 'Random 50%',
                    data: baseline,
                    borderColor: 'rgba(255, 255, 255, 0.35)',
                    borderWidth: 1,
                    pointRadius: 0,
                    borderDash: [4, 4],
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#7a8597' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: {
                    min: 0,
                    max: 1,
                    ticks: { color: '#7a8597', callback: v => `${Math.round(v * 100)}%` },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });
}

function buildPowerRows(metrics) {
    const rows = [
        { label: 'Directional Accuracy', value: metrics.accuracy },
        { label: 'Balanced Accuracy', value: metrics.balanced_accuracy || metrics.accuracy },
        { label: 'Precision', value: metrics.precision },
        { label: 'Recall', value: metrics.recall },
        { label: 'AUC', value: metrics.auc }
    ];

    return rows.map(row => {
        const percent = row.value === null ? '--' : Math.round(row.value * 100);
        return `
            <div class="power-row">
                <div class="power-label">${row.label}</div>
                <div class="bar"><div class="bar-fill" style="width: ${percent === '--' ? 0 : percent}%;"></div></div>
                <div class="text-muted">${percent === '--' ? '--' : `${percent}%`}</div>
            </div>
        `;
    }).join('');
}

function buildActionPill(symbol, confidence, direction) {
    const conf = Math.round((confidence || 0) * 100);
    const cls = direction === 'DOWN' ? 'action-pill down' : 'action-pill';
    return `<span class=\"${cls}\">${symbol} ${conf}%</span>`;
}

function createIndicatorCard(id, item) {
    const div = document.createElement('div');
    div.className = 'indicator-card';

    const changeClass = item.change > 0 ? 'text-green' : item.change < 0 ? 'text-red' : 'text-muted';
    div.innerHTML = `
        <div class="indicator-meta">
            <span>${item.name}</span>
            <span>${item.date}</span>
        </div>
        <div class="indicator-value">${formatValue(item.value, item.unit)}</div>
        <div class="indicator-change ${changeClass}">${formatDelta(item.change, item.unit)} | 1Y ${formatPercent(item.roc_1y)}</div>
        <div class="chart-container">
            <canvas id="chart-${id}"></canvas>
        </div>
    `;

    setTimeout(() => loadChart(id), 50);
    return div;
}

// --- Charts ---

async function loadChart(seriesId) {
    try {
        const res = await fetch(`/api/series/${seriesId}?range=5y`);
        const data = await res.json();

        const ctx = document.getElementById(`chart-${seriesId}`);
        if (!ctx) return;

        const labels = data.data.map(d => d.date);
        const values = data.data.map(d => d.value);
        const start = values[0];
        const end = values[values.length - 1];
        const color = end >= start ? '#2cff9f' : '#f25f5c';

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    borderColor: color,
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: {
                        target: 'origin',
                        above: hexToRgba(color, 0.15),
                        below: hexToRgba(color, 0.15)
                    },
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
                scales: { x: { display: false }, y: { display: false } },
                interaction: { mode: 'nearest', axis: 'x', intersect: false }
            }
        });

    } catch (e) { console.error(e); }
}

async function loadRailChart() {
    try {
        const res = await fetch('/api/series/SP500?range=5y');
        const data = await res.json();
        const ctx = document.getElementById('rail-chart');
        if (!ctx) return;

        const labels = data.data.map(d => d.date);
        const values = data.data.map(d => d.value);
        const color = '#2cff9f';

        if (railChartInstance) {
            railChartInstance.destroy();
        }

        railChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    data: values,
                    borderColor: color,
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: {
                        target: 'origin',
                        above: hexToRgba(color, 0.15),
                        below: hexToRgba(color, 0.15)
                    },
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { x: { display: false }, y: { display: false } }
            }
        });
    } catch (e) { console.error(e); }
}

// --- Helpers ---

function getSeriesItem(category, seriesId) {
    if (!dashboardData[category] || !dashboardData[category].data) return null;
    return dashboardData[category].data[seriesId] || null;
}

function formatValue(val, unit) {
    if (val === null || val === undefined) return '--';
    if (unit === '$B') return `$${Number(val).toLocaleString()}B`;
    if (unit === '$M') return `$${Number(val).toLocaleString()}M`;
    if (unit === '$') return `$${Number(val).toFixed(2)}`;
    if (unit === '%') return `${Number(val).toFixed(2)}%`;
    return Number(val).toLocaleString();
}

function formatDelta(val, unit) {
    if (val === null || val === undefined) return '--';
    const sign = val > 0 ? '+' : '';
    if (unit === '$') return `${sign}$${Number(val).toFixed(2)}`;
    if (unit === '%') return `${sign}${Number(val).toFixed(2)}%`;
    return `${sign}${Number(val).toLocaleString()}`;
}

function formatPercent(val) {
    if (val === null || val === undefined || Number.isNaN(val)) return '--';
    return `${(Number(val) * 100).toFixed(1)}%`;
}

function formatCurrency(val) {
    if (val === null || val === undefined || Number.isNaN(val)) return '--';
    const abs = Math.abs(val);
    const sign = val < 0 ? '-' : '';
    if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(2)}M`;
    if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(1)}K`;
    return `${sign}$${abs.toFixed(0)}`;
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function getHorizonOrder(horizonMap) {
    const map = horizonMap || predictionData?.horizons || {};
    return Object.entries(map)
        .sort((a, b) => (a[1] || 0) - (b[1] || 0))
        .map(([key]) => key);
}

function average(values) {
    if (!values.length) return 0;
    const sum = values.reduce((acc, val) => acc + val, 0);
    return sum / values.length;
}

function startClock() {
    const clock = document.getElementById('clock');
    if (!clock) return;
    const update = () => {
        const now = new Date();
        clock.textContent = now.toLocaleTimeString('en-US', { hour12: false });
    };
    update();
    setInterval(update, 1000);
}
