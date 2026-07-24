/* ==========================================================================
   QUANT ALPHA TERMINAL — COMMAND CENTER JS ENGINE
   100% Real Dynamic Binding — Zero Hardcoded Stubs
   ========================================================================== */

const API = (window.location.protocol === 'file:' || !window.location.host) 
  ? 'http://127.0.0.1:8000/api/v1' 
  : '/api/v1';

let activeStrat = 'v4';
let activeFeedCategory = 'ALL';
let telemetryCache = null;

function init() {
  initClock();
  initCyberSphere();
  initBrainGraphic();

  fetchCommandCenterTelemetry();
  setInterval(fetchCommandCenterTelemetry, 3000);
  setInterval(initClock, 1000);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

function initClock() {
  const now = new Date();
  const utcStr = now.toISOString().substring(11, 19) + ' UTC';
  const el = document.getElementById('mission-clock');
  if (el) el.innerText = utcStr;
}

function onStrategyChange() {
  const selector = document.getElementById('strategy-selector');
  if (selector) {
    activeStrat = selector.value;
    fetchCommandCenterTelemetry();
  }
}

function switchNavTab(tabName, el) {
  document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
  if (el) el.classList.add('active');
}

async function fetchCommandCenterTelemetry() {
  try {
    const res = await fetch(`${API}/command_center/${activeStrat}`);
    if (!res.ok) {
      console.error('API Error:', res.statusText);
      return;
    }
    const data = await res.json();
    telemetryCache = data;

    updateHeader(data);
    updateHealthAndHydro(data.strategy_health, data.market_regime);
    updateTimeline(data.pipeline_steps || []);
    updateCopilotInsights(data.ai_copilot_insights || []);
    renderRiskRadarCanvas(data.risk_radar);
    renderCapitalDonutCanvas(data.capital_allocation);
    updateEventFeed(data.event_feed || []);
    renderFooterSparklines(data);
  } catch (err) {
    console.error('Command Center Telemetry Fetch Error:', err);
  }
}

function updateHeader(data) {
  const summary = data.analytics_summary || {};
  const btcPrice = data.btc_price;
  const btcDelta = data.btc_price_delta_pct;
  const ethPrice = data.eth_price;
  const ethDelta = data.eth_price_delta_pct;

  const btcPriceEl = document.getElementById('hud-btc-price');
  if (btcPriceEl) {
    if (btcPrice !== null && btcPrice !== undefined) {
      const deltaHtml = btcDelta !== null && btcDelta !== undefined 
        ? `<span class="${btcDelta >= 0 ? 'delta-green' : 'delta-red'}">${btcDelta >= 0 ? '+' : ''}${btcDelta.toFixed(2)}%</span>` 
        : '';
      btcPriceEl.innerHTML = `$${btcPrice.toLocaleString('en-US', {minimumFractionDigits: 2})} ${deltaHtml}`;
    } else {
      btcPriceEl.innerText = 'OFFLINE';
    }
  }

  const footBtc = document.getElementById('foot-btc-val');
  if (footBtc) {
    footBtc.innerText = btcPrice ? `$${btcPrice.toLocaleString('en-US', {minimumFractionDigits: 2})}` : '--';
  }
  const footBtcDelta = document.getElementById('foot-btc-delta');
  if (footBtcDelta) {
    footBtcDelta.innerText = btcDelta !== null && btcDelta !== undefined ? `${btcDelta >= 0 ? '+' : ''}${btcDelta.toFixed(2)}%` : '--';
    footBtcDelta.className = `ticker-delta ${btcDelta >= 0 ? 'delta-green' : 'delta-red'}`;
  }

  const footEth = document.getElementById('foot-eth-val');
  if (footEth) {
    footEth.innerText = ethPrice ? `$${ethPrice.toLocaleString('en-US', {minimumFractionDigits: 2})}` : '--';
  }
  const footEthDelta = document.getElementById('foot-eth-delta');
  if (footEthDelta) {
    footEthDelta.innerText = ethDelta !== null && ethDelta !== undefined ? `${ethDelta >= 0 ? '+' : ''}${ethDelta.toFixed(2)}%` : '--';
    footEthDelta.className = `ticker-delta ${ethDelta >= 0 ? 'delta-green' : 'delta-red'}`;
  }

  const equityEl = document.getElementById('hud-equity');
  if (equityEl && summary.current_equity !== undefined) {
    const eqDelta = summary.equity_delta_pct || 0;
    const deltaHtml = `<span class="${eqDelta >= 0 ? 'delta-green' : 'delta-red'}">${eqDelta >= 0 ? '+' : ''}${eqDelta.toFixed(2)}%</span>`;
    equityEl.innerHTML = `$${summary.current_equity.toLocaleString('en-US', {minimumFractionDigits: 2})} ${deltaHtml}`;
  }

  if (summary.net_realized_pnl !== undefined) {
    const pnlEl = document.getElementById('foot-pnl-val');
    if (pnlEl) pnlEl.innerText = `$${summary.net_realized_pnl.toFixed(2)}`;
  }
  if (summary.win_rate !== undefined) {
    const winEl = document.getElementById('foot-winrate-val');
    if (winEl) winEl.innerText = `${summary.win_rate}%`;
  }
  if (summary.total_trades !== undefined) {
    const tradesEl = document.getElementById('foot-trades-val');
    if (tradesEl) tradesEl.innerText = summary.total_trades;
  }

  const statusText = document.getElementById('system-status-text');
  if (statusText) statusText.innerText = 'ALL SYSTEMS NOMINAL';
}

function updateHealthAndHydro(health, regime) {
  if (!health) return;

  const healthPct = health.core_stability_pct !== undefined ? health.core_stability_pct : 0;
  const healthPctEl = document.getElementById('health-pct');
  if (healthPctEl) healthPctEl.innerText = `${Math.round(healthPct)}%`;

  const healthStatusEl = document.getElementById('health-status');
  if (healthStatusEl) healthStatusEl.innerText = health.status || 'NO DATA';

  const circle = document.getElementById('radial-health-circle');
  if (circle) {
    const offset = 264 - (264 * (healthPct / 100));
    circle.style.strokeDashoffset = offset;
  }

  const confPct = health.confidence_level_pct !== undefined ? health.confidence_level_pct : 0;
  const confPctEl = document.getElementById('confidence-pct');
  if (confPctEl) confPctEl.innerText = `${Math.round(confPct)}%`;

  const liquid = document.getElementById('hydro-liquid-fill');
  if (liquid) {
    liquid.style.height = `${confPct}%`;
  }

  if (regime) {
    const regimeName = document.getElementById('regime-name');
    if (regimeName) regimeName.innerText = regime.regime || 'UNAVAILABLE';
    const regimeVol = document.getElementById('regime-vol');
    if (regimeVol) regimeVol.innerText = regime.volatility || 'UNAVAILABLE';
    const regimeLiq = document.getElementById('regime-liq');
    if (regimeLiq) regimeLiq.innerText = regime.liquidity || 'UNAVAILABLE';
  }
}

function updateTimeline(steps) {
  const container = document.getElementById('timeline-nodes-container');
  if (!container) return;

  if (!steps || steps.length === 0) {
    container.innerHTML = '<div class="timeline-empty-msg" style="font-size: 0.65rem; color: #64748b; padding: 10px; text-align: center; width: 100%;">No active execution pipeline trace recorded yet. Click RUN BOT EXECUTION to generate steps.</div>';
    return;
  }

  const icons = ['⚡', '🛡️', '📏', '📤', '✅', '💼', '🔒', '🎯', '📈'];

  container.innerHTML = steps.map((s, idx) => `
    <div class="pipeline-node-item">
      <div class="node-icon-box" style="${s.status === 'DONE' ? 'border-color: #00f5d4; color: #00f5d4;' : 'opacity: 0.4;'}">${icons[idx % icons.length]}</div>
      <div class="node-title-text">${s.label}</div>
      <div class="node-timestamp">${s.time}</div>
    </div>
  `).join('');
}

function updateCopilotInsights(insights) {
  const container = document.getElementById('copilot-insights-list');
  if (!container) return;

  if (!insights || insights.length === 0) {
    container.innerHTML = '<div class="copilot-item"><span>No AI insights generated yet.</span></div>';
    return;
  }

  container.innerHTML = insights.map(i => `
    <div class="copilot-item ${i.highlight ? 'highlight' : ''}">
      <span class="copilot-time">${i.time}</span>
      <span>${i.text}</span>
    </div>
  `).join('');
}

/* ==========================================================================
   PURE HTML5 CANVAS RISK RADAR CHART RENDERER
   ========================================================================== */
function renderRiskRadarCanvas(riskData) {
  if (!riskData) return;

  if (riskData.max_drawdown_pct) document.getElementById('risk-max-dd').innerText = riskData.max_drawdown_pct;
  if (riskData.var_95_pct) document.getElementById('risk-var95').innerText = riskData.var_95_pct;
  if (riskData.exposure_pct) document.getElementById('risk-exposure').innerText = riskData.exposure_pct;
  if (riskData.leverage) document.getElementById('risk-leverage').innerText = riskData.leverage;

  const canvas = document.getElementById('chart-risk-radar');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const width = canvas.parentElement.clientWidth || 280;
  const height = canvas.parentElement.clientHeight || 160;
  canvas.width = width;
  canvas.height = height;

  ctx.clearRect(0, 0, width, height);

  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(centerX, centerY) - 22;
  const numAxes = 5;
  const labels = ['Drawdown', 'Exposure', 'Leverage', 'Liquidity', 'Volatility'];
  
  const scoresObj = riskData.radar_scores || { drawdown: 0, exposure: 0, leverage: 0, liquidity: 0, volatility: 0 };
  const values = [
    scoresObj.drawdown || 0,
    scoresObj.exposure || 0,
    scoresObj.leverage || 0,
    scoresObj.liquidity || 0,
    scoresObj.volatility || 0
  ];

  // Draw Pentagon Grid Rings
  const rings = 4;
  for (let r = 1; r <= rings; r++) {
    const rRadius = (radius / rings) * r;
    ctx.beginPath();
    for (let i = 0; i < numAxes; i++) {
      const angle = (Math.PI * 2 / numAxes) * i - Math.PI / 2;
      const x = centerX + rRadius * Math.cos(angle);
      const y = centerY + rRadius * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = 'rgba(0, 245, 212, 0.15)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Draw Axes & Axis Labels
  ctx.font = '9px Inter, sans-serif';
  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (let i = 0; i < numAxes; i++) {
    const angle = (Math.PI * 2 / numAxes) * i - Math.PI / 2;
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);

    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(x, y);
    ctx.strokeStyle = 'rgba(0, 245, 212, 0.2)';
    ctx.stroke();

    const labelX = centerX + (radius + 12) * Math.cos(angle);
    const labelY = centerY + (radius + 12) * Math.sin(angle);
    ctx.fillText(labels[i], labelX, labelY);
  }

  // Draw Filled Risk Polygon
  ctx.beginPath();
  for (let i = 0; i < numAxes; i++) {
    const valRatio = (values[i] / 100.0);
    const angle = (Math.PI * 2 / numAxes) * i - Math.PI / 2;
    const x = centerX + radius * valRatio * Math.cos(angle);
    const y = centerY + radius * valRatio * Math.sin(angle);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();

  ctx.fillStyle = 'rgba(0, 245, 212, 0.22)';
  ctx.fill();
  ctx.strokeStyle = '#00f5d4';
  ctx.lineWidth = 2;
  ctx.shadowColor = '#00f5d4';
  ctx.shadowBlur = 8;
  ctx.stroke();
  ctx.shadowBlur = 0;
}

/* ==========================================================================
   PURE HTML5 CANVAS CAPITAL ALLOCATION DONUT RENDERER
   ========================================================================== */
function renderCapitalDonutCanvas(capital) {
  if (!capital) return;

  const totalValEl = document.getElementById('donut-total-val');
  if (totalValEl && capital.total_equity !== undefined) {
    totalValEl.innerText = `$${capital.total_equity.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
  }

  const legendList = document.getElementById('donut-legend-list');
  if (legendList && capital.assets) {
    legendList.innerHTML = capital.assets.map(a => `
      <div class="legend-item">
        <div class="legend-left">
          <div class="legend-dot" style="background: ${a.color};"></div>
          <span>${a.name}</span>
        </div>
        <strong style="color: #fff;">${a.pct}%</strong>
      </div>
    `).join('');
  }

  const canvas = document.getElementById('chart-capital-donut');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const size = 100;
  canvas.width = size;
  canvas.height = size;

  ctx.clearRect(0, 0, size, size);

  const centerX = size / 2;
  const centerY = size / 2;
  const outerRadius = 46;
  const innerRadius = 32;

  let startAngle = -Math.PI / 2;

  const assets = capital.assets || [];

  if (assets.length === 0) {
    ctx.beginPath();
    ctx.arc(centerX, centerY, outerRadius, 0, Math.PI * 2, false);
    ctx.arc(centerX, centerY, innerRadius, Math.PI * 2, 0, true);
    ctx.closePath();
    ctx.fillStyle = '#1e293b';
    ctx.fill();
    return;
  }

  assets.forEach(a => {
    const sliceAngle = (a.pct / 100.0) * (Math.PI * 2);
    const endAngle = startAngle + sliceAngle;

    ctx.beginPath();
    ctx.arc(centerX, centerY, outerRadius, startAngle, endAngle, false);
    ctx.arc(centerX, centerY, innerRadius, endAngle, startAngle, true);
    ctx.closePath();

    ctx.fillStyle = a.color;
    ctx.fill();

    startAngle = endAngle;
  });
}

function updateEventFeed(events) {
  const container = document.getElementById('event-feed-list');
  if (!container) return;

  if (!events || events.length === 0) {
    container.innerHTML = '<div style="font-size: 0.65rem; color: #64748b; padding: 10px; text-align: center;">No event logs recorded in database.</div>';
    return;
  }

  const filtered = activeFeedCategory === 'ALL'
    ? events
    : events.filter(e => e.category === activeFeedCategory);

  if (filtered.length === 0) {
    container.innerHTML = `<div style="font-size: 0.65rem; color: #64748b; padding: 10px; text-align: center;">No ${activeFeedCategory} events found.</div>`;
    return;
  }

  container.innerHTML = filtered.map(e => `
    <div class="feed-row">
      <span class="feed-time">${e.time}</span>
      <span class="feed-cat-badge cat-${e.category}">● ${e.category}</span>
      <span class="feed-msg">${e.message}</span>
    </div>
  `).join('');
}

function filterFeed(category, btn) {
  activeFeedCategory = category;
  document.querySelectorAll('.feed-tab-btn').forEach(b => b.classList.remove('active-tab-block'));
  if (btn) btn.classList.add('active-tab-block');
  if (telemetryCache && telemetryCache.event_feed) updateEventFeed(telemetryCache.event_feed);
}

/* ==========================================================================
   FOOTER SPARKLINE CANVAS RENDERERS
   ========================================================================== */
function renderFooterSparklines(data) {
  if (!data || !data.sparkline_data) return;

  const sp = data.sparkline_data;
  const sparkConfigs = [
    { id: 'spark-btc', color: '#00f5a0', points: sp.btc_prices || [] },
    { id: 'spark-eth', color: '#00d2ff', points: sp.eth_prices || [] },
    { id: 'spark-pnl', color: '#00f5a0', points: sp.pnl_cumulative || [] },
    { id: 'spark-win', color: '#a855f7', points: sp.win_rate_rolling || [] }
  ];

  sparkConfigs.forEach(cfg => {
    const canvas = document.getElementById(cfg.id);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = 50;
    canvas.height = 18;

    ctx.clearRect(0, 0, 50, 18);

    if (!cfg.points || cfg.points.length < 2) return;

    const min = Math.min(...cfg.points);
    const max = Math.max(...cfg.points);
    const range = max - min || 1;

    ctx.beginPath();
    cfg.points.forEach((pt, i) => {
      const x = (i / (cfg.points.length - 1)) * 50;
      const y = 18 - ((pt - min) / range) * 14 - 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.strokeStyle = cfg.color;
    ctx.lineWidth = 1.5;
    ctx.shadowColor = cfg.color;
    ctx.shadowBlur = 4;
    ctx.stroke();
    ctx.shadowBlur = 0;
  });
}

/* ==========================================================================
   QUANT ALPHA CORE 3D PARTICLE GLOBE CANVAS RENDERER
   ========================================================================== */
function initCyberSphere() {
  const wrapper = document.querySelector('.cyber-sphere-wrapper');
  if (!wrapper) return;

  let canvas = document.getElementById('cyber-particle-canvas');
  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.id = 'cyber-particle-canvas';
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    wrapper.appendChild(canvas);
  }

  const ctx = canvas.getContext('2d');
  const width = 240;
  const height = 240;
  canvas.width = width;
  canvas.height = height;

  const numParticles = 65;
  const particles = [];
  const radius = 62;

  for (let i = 0; i < numParticles; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos((Math.random() * 2) - 1);
    particles.push({
      x: radius * Math.sin(phi) * Math.cos(theta),
      y: radius * Math.sin(phi) * Math.sin(theta),
      z: radius * Math.cos(phi)
    });
  }

  let angleY = 0;
  let angleX = 0;

  function animate() {
    ctx.clearRect(0, 0, width, height);
    const cx = width / 2;
    const cy = height / 2;

    angleY += 0.01;
    angleX += 0.005;

    const projected = [];

    particles.forEach(p => {
      let x1 = p.x * Math.cos(angleY) - p.z * Math.sin(angleY);
      let z1 = p.x * Math.sin(angleY) + p.z * Math.cos(angleY);
      let y2 = p.y * Math.cos(angleX) - z1 * Math.sin(angleX);
      let z2 = p.y * Math.sin(angleX) + z1 * Math.cos(angleX);

      const scale = 200 / (200 + z2);
      const px = cx + x1 * scale;
      const py = cy + y2 * scale;

      projected.push({ x: px, y: py, z: z2 });
    });

    // Draw lines between close particles
    ctx.strokeStyle = 'rgba(0, 245, 212, 0.22)';
    ctx.lineWidth = 0.8;
    for (let i = 0; i < projected.length; i++) {
      for (let j = i + 1; j < projected.length; j++) {
        const dx = projected[i].x - projected[j].x;
        const dy = projected[i].y - projected[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 36) {
          ctx.beginPath();
          ctx.moveTo(projected[i].x, projected[i].y);
          ctx.lineTo(projected[j].x, projected[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw Particle Nodes
    projected.forEach(p => {
      const pSize = Math.max(1, (p.z + radius) / 35);
      ctx.beginPath();
      ctx.arc(p.x, p.y, pSize, 0, Math.PI * 2);
      ctx.fillStyle = p.z > 0 ? '#00f5d4' : '#00b4d8';
      ctx.shadowColor = '#00f5d4';
      ctx.shadowBlur = 4;
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    requestAnimationFrame(animate);
  }

  animate();
}

/* ==========================================================================
   AI COPILOT NEURAL BRAIN NETWORK GRAPHIC RENDERER
   ========================================================================== */
function initBrainGraphic() {
  const container = document.querySelector('.copilot-brain-graphic');
  if (!container) return;

  container.innerHTML = `
    <svg class="brain-svg" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M20 50 Q 35 20, 50 30 T 80 50" stroke="#00d2ff" stroke-width="1.5" opacity="0.7" stroke-dasharray="3 3"/>
      <path d="M20 50 Q 35 80, 50 70 T 80 50" stroke="#00f5d4" stroke-width="1.5" opacity="0.7"/>
      <path d="M50 30 Q 60 50, 50 70" stroke="#a855f7" stroke-width="1.5" opacity="0.6"/>
      <path d="M30 35 Q 50 50, 70 35" stroke="#00f5d4" stroke-width="1.2" opacity="0.6"/>

      <circle cx="20" cy="50" r="4" fill="#00d2ff"/>
      <circle cx="35" cy="25" r="3" fill="#00f5d4"/>
      <circle cx="65" cy="25" r="3" fill="#00f5d4"/>
      <circle cx="80" cy="50" r="4" fill="#00d2ff"/>
      <circle cx="65" cy="75" r="3" fill="#a855f7"/>
      <circle cx="35" cy="75" r="3" fill="#a855f7"/>
      <circle cx="50" cy="50" r="5" fill="#00f5d4"/>
    </svg>
  `;
}

async function triggerExecution() {
  try {
    const res = await fetch(`${API}/bot/run?strategy_id=${activeStrat}`, { method: 'POST' });
    const data = await res.json();
    fetchCommandCenterTelemetry();
  } catch (e) {
    console.error('Execution Error:', e);
  }
}
