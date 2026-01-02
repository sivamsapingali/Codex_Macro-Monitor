// State
let dashboardData = {};
let intelligenceData = {};
let predictionData = {};
let usageData = {};
let currentCategory = 'growth';
let railChartInstance = null;
let methodChartInstance = null;

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

    setInterval(async () => {
        await Promise.all([
            loadIntelligence(),
            loadDashboard(),
            loadPredictions(),
            loadUsage()
        ]);
    }, 60000);
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
        if (statusMeta) statusMeta.textContent = `${data.data_points || '--'} series`;
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
        renderMethodology();
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

        const horizonOrder = ['1w', '1m', '3m', '6m', '12m'];
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

    const horizons = ['1m', '3m'];
    container.innerHTML = '';

    horizons.forEach(horizon => {
        const ranked = predictionData.ranked[horizon];
        if (!ranked) return;

        const bullish = ranked.bullish || [];
        const bearish = ranked.bearish || [];
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
            powerEl.innerHTML = `${rows}<div class="text-muted">${asOf}</div>`;
        }
    }

    if (summaryEl) {
        const stats = computeHorizonStats();
        if (!stats) {
            summaryEl.textContent = 'Waiting on backtest stats...';
        } else {
            const avgAcc = average(Object.values(stats).map(s => s.accuracy));
            const avgEdge = avgAcc - 0.5;
            summaryEl.textContent = `Average out-of-sample accuracy: ${(avgAcc * 100).toFixed(1)}% (edge ${(avgEdge * 100).toFixed(1)}% over random).`;
            renderMethodologyChart(stats);
        }
    }
}

function collectModelMetrics() {
    if (!predictionData || !predictionData.predictions) return null;

    const horizons = Object.keys(predictionData.horizons || {});
    let acc = 0;
    let precision = 0;
    let recall = 0;
    let count = 0;

    Object.values(predictionData.predictions).forEach(sector => {
        horizons.forEach(h => {
            const metrics = sector.horizons?.[h]?.metrics;
            if (!metrics) return;
            acc += metrics.accuracy || 0;
            precision += metrics.precision || 0;
            recall += metrics.recall || 0;
            count += 1;
        });
    });

    if (!count) return null;
    return {
        accuracy: acc / count,
        precision: precision / count,
        recall: recall / count,
        samples: count
    };
}

function computeHorizonStats() {
    if (!predictionData || !predictionData.predictions || !predictionData.horizons) return null;
    const horizons = Object.keys(predictionData.horizons);
    const stats = {};

    horizons.forEach(h => {
        stats[h] = { accuracy: 0, precision: 0, recall: 0, count: 0 };
    });

    Object.values(predictionData.predictions).forEach(sector => {
        horizons.forEach(h => {
            const metrics = sector.horizons?.[h]?.metrics;
            if (!metrics) return;
            stats[h].accuracy += metrics.accuracy || 0;
            stats[h].precision += metrics.precision || 0;
            stats[h].recall += metrics.recall || 0;
            stats[h].count += 1;
        });
    });

    horizons.forEach(h => {
        if (stats[h].count) {
            stats[h].accuracy /= stats[h].count;
            stats[h].precision /= stats[h].count;
            stats[h].recall /= stats[h].count;
        }
    });

    return stats;
}

function renderMethodologyChart(stats) {
    const ctx = document.getElementById('method-accuracy-chart');
    if (!ctx) return;
    const labels = Object.keys(stats);
    const accuracy = labels.map(label => stats[label].accuracy || 0);
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
                    label: 'OOS Accuracy',
                    data: accuracy,
                    backgroundColor: 'rgba(44, 255, 159, 0.4)',
                    borderColor: 'rgba(44, 255, 159, 0.8)',
                    borderWidth: 1,
                },
                {
                    type: 'line',
                    label: 'Random 50%',
                    data: baseline,
                    borderColor: 'rgba(255, 255, 255, 0.4)',
                    borderWidth: 1,
                    pointRadius: 0,
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
        { label: 'Precision', value: metrics.precision },
        { label: 'Recall', value: metrics.recall }
    ];

    return rows.map(row => {
        const percent = Math.round(row.value * 100);
        return `
            <div class="power-row">
                <div class="power-label">${row.label}</div>
                <div class="bar"><div class="bar-fill" style="width: ${percent}%;"></div></div>
                <div class="text-muted">${percent}%</div>
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

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
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
