const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (v, d=1) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : 'NA';
const pct = (v, d=1) => Number.isFinite(Number(v)) ? `${(Number(v)*100).toFixed(d)}%` : 'NA';
const clamp = (v, lo=0, hi=100) => Math.max(lo, Math.min(hi, Number(v) || 0));

let bundle = null;
let activeIndex = null;
let activeMeta = 'overview';
let activeChart = 'ridge';
let chart = null;

function stateClass(state) {
  return `state-pill state-${esc(state || 'NA')}`;
}

function scoreClass(v) {
  return v >= 70 ? 'bad' : v >= 45 ? 'warn' : 'good';
}

function idx() {
  return (bundle?.indices || []).find(x => x.key === activeIndex) || (bundle?.indices || [])[0];
}

function dataSeries() {
  const i = idx();
  return i ? (($("viewSelect")?.value || 'recent') === 'long' ? i.series_long : i.series_recent) || [] : [];
}

function targetLabel(key) {
  return [...(bundle.targets?.crash || []), ...(bundle.targets?.bull || [])].find(t => t.key === key)?.label || key || 'target';
}

function bar(value, kind) {
  const v = clamp(value);
  const cls = kind || scoreClass(v);
  return `<div class="bar ${cls}"><i style="width:${v}%"></i></div>`;
}

function isBullTarget() {
  return String($('targetSelect')?.value || '').startsWith('up_');
}

async function boot() {
  bindStaticEvents();

  try {
    const res = await fetch('data/derived/market_ridge_radar.json', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    bundle = await res.json();

    initializeControls();

    $('notice').className = bundle.metadata?.mode === 'synthetic_demo'
      ? 'notice panel-lite warn'
      : 'notice panel-lite ok';

    $('notice').innerHTML = bundle.metadata?.mode === 'synthetic_demo'
      ? 'Demo-mode bundle loaded. This should not appear on the live site.'
      : 'Live public-history bundle loaded. All values are research diagnostics, not trading instructions.';

    renderAll();
  } catch (err) {
    $('notice').className = 'notice panel-lite bad';
    $('notice').textContent = `Could not load app data: ${err.message}`;
  }
}

function bindStaticEvents() {
  $('themeBtn').addEventListener('click', () => {
    document.documentElement.classList.toggle('light');
    $('themeBtn').textContent = document.documentElement.classList.contains('light') ? 'Light' : 'Dark';
    drawChart();
  });

  document.querySelectorAll('.meta-tab').forEach(btn => btn.addEventListener('click', () => {
    document.querySelectorAll('.meta-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    activeMeta = btn.dataset.meta;

    if (activeMeta === 'dustlambda') activeChart = 'lambda';
    else if (activeMeta === 'scorecard') activeChart = 'score';
    else if (activeMeta === 'audit') activeChart = 'lambda';
    else activeChart = 'ridge';

    syncChartTabs();
    renderAll();
  }));

  document.querySelectorAll('.chart-tab').forEach(btn => btn.addEventListener('click', () => {
    document.querySelectorAll('.chart-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    activeChart = btn.dataset.chart;
    drawChart();
  }));
}

function syncChartTabs() {
  document.querySelectorAll('.chart-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.chart === activeChart);
  });
}

function initializeControls() {
  const indices = bundle.indices || [];
  activeIndex = activeIndex || indices[0]?.key;

  $('indexSelect').innerHTML = indices
    .map(i => `<option value="${esc(i.key)}">${esc(i.key)} - ${esc(i.name)}</option>`)
    .join('');

  $('indexSelect').value = activeIndex;

  const targets = [...(bundle.targets?.crash || []), ...(bundle.targets?.bull || [])];

  $('targetSelect').innerHTML = targets
    .map(t => `<option value="${esc(t.key)}">${esc(t.label)}</option>`)
    .join('');

  $('targetSelect').value = targets[1]?.key || targets[0]?.key || '';

  ['indexSelect', 'viewSelect', 'targetSelect'].forEach(id => {
    $(id).addEventListener('change', () => {
      if (id === 'indexSelect') activeIndex = $(id).value;
      renderAll();
    });
  });
}

function renderAll() {
  if (!bundle) return;

  const m = bundle.metadata || {};
  const s = bundle.summary || {};
  const current = idx()?.current || {};

  $('generatedAt').textContent = (m.generated_at || '').slice(0, 16).replace('T', ' ') || 'NA';
  $('bundleMode').textContent = m.mode || 'strict';
  $('globalState').textContent = s.global_state || 'NA';

  $('selectedState').className = stateClass(current.state);
  $('selectedState').textContent = current.state || 'NA';
  $('selectedName').textContent = `${idx()?.key || ''} - ${idx()?.name || ''}`;

  renderKpis();
  renderCurrentRead();
  renderNarrative();
  renderWorkspace();
  drawChart();
}

function renderKpis() {
  const s = bundle.summary || {};
  const c = idx()?.current || {};

  const cards = [
    ['Global state', s.global_state || 'NA', `${s.red_count || 0} red / ${s.yellow_count || 0} yellow / ${s.green_count || 0} green`],
    ['Selected risk', `${num(c.risk_score, 1)}`, `${num(c.risk_pct, 0)}th percentile`],
    ['Ridge failure', `${num(c.ridge_failure_score, 1)}`, `${c.ridge_state || 'NA'} / ${num(c.ridge_failure_pct, 0)}th pct`],
    ['Dust pressure', `${num(c.first_notice_score, 1)}`, `${c.dust_state || 'NA'} / dust z ${num(c.dust_z, 2)}`],
    ['Bull inertia', `${num(c.bull_score, 1)}`, `raw ${num(c.bull_raw_score, 1)}; gate ${num(c.crash_gate, 2)}`],
    ['lambda_q flicker', `${num(c.lambda_flicker_score, 1)}`, `lambda_q ${num(c.lambda_q, 1)} / beta ${num(c.beta, 2)}`],
    ['Score target', targetLabel($('targetSelect').value), isBullTarget() ? 'bull-run target mode' : 'drawdown target mode']
  ];

  $('kpis').innerHTML = cards
    .map(([a, b, c]) => `<div class="kpi-card"><span>${esc(a)}</span><b>${esc(b)}</b><em>${esc(c)}</em></div>`)
    .join('');
}

function renderCurrentRead() {
  const i = idx();
  const c = i?.current || {};

  const tiles = [
    ['Event', `${c.event_state || c.state || 'NA'}`, `composite ${num(c.risk_score, 1)} / pct ${num(c.risk_pct, 0)}`],
    ['Ridge', `${c.ridge_state || 'NA'}`, `failure ${num(c.ridge_failure_score, 1)} / pct ${num(c.ridge_failure_pct, 0)}`],
    ['Dust', `${c.dust_state || 'NA'}`, `first-notice ${num(c.first_notice_score, 1)} / dust z ${num(c.dust_z, 2)}`],
    ['Pullback', `${num(c.excursion, 2)} sigma`, 'persistent envelope excursion warns of failure'],
    ['S2 fit', `R2 ${num(c.s2_r2, 3)}`, `Delta BIC vs D=1 ${num(c.delta_bic_vs_d1, 1)}`],
    ['Bull inertia', `${num(c.bull_score, 1)} / 100`, `raw ${num(c.bull_raw_score, 1)}; gate ${num(c.crash_gate, 2)}`]
  ];

  $('currentRead').innerHTML = tiles
    .map(t => `<div class="metric-tile"><span>${esc(t[0])}</span><b>${esc(t[1])}</b><p>${esc(t[2])}</p></div>`)
    .join('');
}

function renderNarrative() {
  const s = bundle.summary || {};
  const i = idx();

  const risk = (s.top_risk || [])
    .map(x => `${x.key} ${num(x.risk_pct ?? x.risk_score, 0)}`)
    .join(' / ') || 'none';

  const bull = (s.top_bull || [])
    .map(x => `${x.key} ${num(x.bull_score, 0)}`)
    .join(' / ') || 'none';

  const modeNote = isBullTarget()
    ? 'Bull target mode: bull inertia is shown with crash-gate suppression. It is not a safety label.'
    : 'Crash target mode: displayed score separates risk percentile, first-notice dust pressure, and ridge-failure percentile.';

  $('globalNarrative').innerHTML =
    `<strong>Narrate.</strong> ${esc(i?.narrative || 'No selected-index narrative.')}<br><br>` +
    `<strong>Mode:</strong> ${esc(modeNote)}<br><br>` +
    `<strong>Global leaders:</strong> risk ${esc(risk)}; bull ${esc(bull)}. ` +
    `Median risk ${num(s.median_risk, 1)}, median first-notice ${num(s.median_first_notice, 1)}, median ridge-failure ${num(s.median_ridge_failure, 1)}.`;
}

function drawChart() {
  const i = idx();
  if (!i) return;

  const data = dataSeries();

  $('chartTitle').textContent =
    `${i.key} ${activeChart === 'ridge' ? 'ridge theater' :
      activeChart === 'dust' ? 'dust cloud' :
      activeChart === 'lambda' ? 'lambda_q flicker' :
      'event scores'}`;

  $('chartSubtitle').textContent =
    activeChart === 'ridge' ? 'close, retained ridge, dust envelope, correction risk, bull inertia' :
    activeChart === 'dust' ? 'residual-cloud thickness, first-notice pressure, ridge excursion' :
    activeChart === 'lambda' ? 'rolling S2 coherence scale and fit instability' :
    isBullTarget()
      ? `${targetLabel($('targetSelect').value)} bull-score context`
      : `${targetLabel($('targetSelect').value)} drawdown-risk context`;

  $('chartNarrative').textContent =
    i.narrative || 'Retained ridge is the low-frequency structure; dust cloud is the operational residual around it; lambda_q flicker marks unstable S2 probe scale.';

  if (!window.echarts) {
    drawFallbackSvg(data);
    return;
  }

  if (!chart) chart = echarts.init($('mainChart'));

  const light = document.documentElement.classList.contains('light');

  const p = light
    ? { text:'#162331', muted:'#617181', grid:'rgba(33,51,68,.14)', price:'#0878bd', ridge:'#14844a', risk:'#c83434', bull:'#168d54', dust:'#9c6700', lambda:'#6741d9', env:'rgba(8,120,189,.12)' }
    : { text:'#edf4f8', muted:'#9aacba', grid:'rgba(179,208,230,.18)', price:'#70c8ff', ridge:'#79e39c', risk:'#ff6b6b', bull:'#9df1c5', dust:'#ffd166', lambda:'#c7a3ff', env:'rgba(112,200,255,.13)' };

  const dates = data.map(d => d.date);

  const axis = {
    type: 'category',
    data: dates,
    axisLine: { lineStyle: { color: p.grid } },
    axisLabel: { color: p.muted, hideOverlap: true, fontSize: 10 }
  };

  const base = {
    animation: false,
    textStyle: { color: p.text, fontSize: 11 },
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { color: p.muted, fontSize: 10 }, itemWidth: 12, itemHeight: 7 },
    grid: { left: 46, right: 36, top: 34, bottom: 26 },
    xAxis: axis
  };

  let option;

  if (activeChart === 'ridge') {
    option = {
      ...base,
      yAxis: [
        { type:'log', axisLabel:{color:p.muted,fontSize:10}, splitLine:{lineStyle:{color:p.grid}} },
        { type:'value', min:0, max:100, axisLabel:{color:p.muted,fontSize:10}, splitLine:{show:false} }
      ],
      series: [
        { name:'Close', type:'line', yAxisIndex:0, data:data.map(d => d.close), showSymbol:false, lineStyle:{width:1.3,color:p.price} },
        { name:'Ridge', type:'line', yAxisIndex:0, data:data.map(d => d.ridge_price), showSymbol:false, lineStyle:{width:1.5,color:p.ridge} },
        { name:'Env hi', type:'line', yAxisIndex:0, data:data.map(d => d.envelope_hi), showSymbol:false, lineStyle:{width:.7,color:p.env}, areaStyle:{color:p.env} },
        { name:'Env lo', type:'line', yAxisIndex:0, data:data.map(d => d.envelope_lo), showSymbol:false, lineStyle:{width:.7,color:p.env} },
        { name:'Risk pct', type:'bar', yAxisIndex:1, data:data.map(d => d.risk_pct ?? d.risk_score), itemStyle:{color:p.risk, opacity:.24} },
        { name:'Bull', type:'bar', yAxisIndex:1, data:data.map(d => d.bull_score), itemStyle:{color:p.bull, opacity:.18} }
      ]
    };
  } else if (activeChart === 'dust') {
    option = {
      ...base,
      yAxis: [
        { type:'value', axisLabel:{color:p.muted,fontSize:10}, splitLine:{lineStyle:{color:p.grid}} },
        { type:'value', min:0, max:100, axisLabel:{color:p.muted,fontSize:10}, splitLine:{show:false} }
      ],
      series: [
        { name:'Dust z', type:'line', data:data.map(d => d.dust_z), showSymbol:false, lineStyle:{width:1.4,color:p.dust} },
        { name:'Dust pct', type:'line', yAxisIndex:1, data:data.map(d => d.dust_pct), showSymbol:false, lineStyle:{width:1.0,color:p.price} },
        { name:'First notice', type:'line', yAxisIndex:1, data:data.map(d => d.first_notice_score), showSymbol:false, lineStyle:{width:1.2,color:p.risk} },
        { name:'Excursion', type:'line', data:data.map(d => d.excursion), showSymbol:false, lineStyle:{width:1.1,color:p.ridge} }
      ]
    };
  } else if (activeChart === 'lambda') {
    option = {
      ...base,
      yAxis: [
        { type:'value', name:'lambda/beta', axisLabel:{color:p.muted,fontSize:10}, splitLine:{lineStyle:{color:p.grid}} },
        { type:'value', name:'flicker', min:0, max:100, axisLabel:{color:p.muted,fontSize:10}, splitLine:{show:false} }
      ],
      series: [
        { name:'lambda_q', type:'line', data:data.map(d => d.lambda_q), showSymbol:false, lineStyle:{width:1.4,color:p.lambda} },
        { name:'beta', type:'line', data:data.map(d => d.beta), showSymbol:false, lineStyle:{width:1.0,color:p.ridge} },
        { name:'Flicker', type:'bar', yAxisIndex:1, data:data.map(d => d.lambda_flicker_score), itemStyle:{color:p.risk, opacity:.24} }
      ]
    };
  } else {
    const scoreSeries = isBullTarget()
      ? [
          { name:'Bull score', type:'line', data:data.map(d => d.bull_score), showSymbol:false, lineStyle:{width:1.5,color:p.bull} },
          { name:'Raw bull', type:'line', data:data.map(d => d.bull_raw_score ?? d.bull_score), showSymbol:false, lineStyle:{width:1.0,color:p.ridge,type:'dashed'} },
          { name:'Crash gate', type:'bar', yAxisIndex:1, data:data.map(d => 100 * (d.crash_gate || 0)), itemStyle:{color:p.risk, opacity:.22} }
        ]
      : [
          { name:'Risk pct', type:'line', data:data.map(d => d.risk_pct ?? d.risk_score), showSymbol:false, lineStyle:{width:1.5,color:p.risk} },
          { name:'First notice pct', type:'line', data:data.map(d => d.first_notice_pct ?? d.first_notice_score), showSymbol:false, lineStyle:{width:1.1,color:p.dust} },
          { name:'Ridge failure pct', type:'line', data:data.map(d => d.ridge_failure_pct ?? d.ridge_failure_score), showSymbol:false, lineStyle:{width:1.1,color:p.lambda} },
          { name:'Drawdown 252', type:'bar', yAxisIndex:1, data:data.map(d => 100 * (d.drawdown_252 || 0)), itemStyle:{color:p.dust, opacity:.18} }
        ];

    option = {
      ...base,
      yAxis: [
        { type:'value', min:0, max:100, axisLabel:{color:p.muted,fontSize:10}, splitLine:{lineStyle:{color:p.grid}} },
        { type:'value', axisLabel:{color:p.muted,fontSize:10}, splitLine:{show:false} }
      ],
      series: scoreSeries
    };
  }

  chart.setOption(option, true);
  chart.resize();
}

function drawFallbackSvg(data) {
  const el = $('mainChart');
  const pts = data.slice(-160).map((d, k) => [k, Number(d.risk_pct ?? d.risk_score) || 0]);
  const w = el.clientWidth || 700;
  const h = el.clientHeight || 280;

  const path = pts
    .map(([x, y], k) => `${k ? 'L' : 'M'} ${20 + x * (w - 40) / Math.max(1, pts.length - 1)} ${h - 20 - y * (h - 40) / 100}`)
    .join(' ');

  el.innerHTML =
    `<svg width="100%" height="100%" viewBox="0 0 ${w} ${h}">` +
    `<rect width="${w}" height="${h}" fill="transparent"/>` +
    `<path d="${path}" fill="none" stroke="currentColor" stroke-width="2"/>` +
    `</svg>`;
}

function renderWorkspace() {
  if (activeMeta === 'overview') return renderOverviewWorkspace();
  if (activeMeta === 'ridge') return renderRidgeWorkspace();
  if (activeMeta === 'dustlambda') return renderDustLambdaWorkspace();
  if (activeMeta === 'scorecard') return renderScorecardWorkspace();
  return renderAuditWorkspace();
}

function panel(title, hint, body) {
  return `<section class="subpanel"><h3>${esc(title)}</h3>${hint ? `<div class="hint">${esc(hint)}</div>` : ''}${body}</section>`;
}

function tableWrap(html) {
  return `<div class="table-wrap">${html}</div>`;
}

function renderOverviewWorkspace() {
  $('workspace').innerHTML = `<div class="workspace-grid">
    ${panel('Global matrix', 'sorted by current risk percentile', tableWrap(matrixTable(8)))}
    ${panel('Selected scorecard', targetLabel($('targetSelect').value), tableWrap(backtestTable(idx()?.key)))}
    ${panel('Crash / bull lab', 'historical stress windows', tableWrap(caseTable(idx()?.key, 8)))}
    ${panel('Narrate', 'single-screen interpretation', overviewStory())}
  </div>`;
}

function renderRidgeWorkspace() {
  $('workspace').innerHTML = `<div class="workspace-grid three">
    ${panel('Ridge-health matrix', 'event, ridge, and dust split', tableWrap(matrixTable(20)))}
    ${panel('Top stress leaders', 'highest risk percentile now', tableWrap(leaderTable('risk')))}
    ${panel('Top bull leaders', 'highest bull-inertia stack now', tableWrap(leaderTable('bull')))}
  </div>`;
}

function renderDustLambdaWorkspace() {
  $('workspace').innerHTML = `<div class="workspace-grid three">
    ${panel('lambda_q tail', 'rolling S2 fit audit for selected index', tableWrap(lambdaTable(24)))}
    ${panel('Dust / first-notice ranking', 'residual cloud and early pressure', tableWrap(dustRankTable()))}
    ${panel('S2 method note', 'what is being measured', methodNote())}
  </div>`;
}

function renderScorecardWorkspace() {
  $('workspace').innerHTML = `<div class="workspace-grid two">
    ${panel('Prior prediction scorecard', targetLabel($('targetSelect').value), tableWrap(backtestTable(null)))}
    ${panel('Crash / bull-run event lab', 'historical stress windows by selected target class', tableWrap(caseTable(null, 50)))}
  </div>`;
}

function renderAuditWorkspace() {
  $('workspace').innerHTML = `<div class="workspace-grid three">
    ${panel('Data health', 'source / fallback audit', tableWrap(healthTable()))}
    ${panel('Selected lambda_q audit', 'latest rolling fits', tableWrap(lambdaTable(32)))}
    ${panel('Caveats', 'research policy', caveats())}
  </div>`;
}

function matrixTable(limit) {
  const rows = [...(bundle.indices || [])]
    .sort((a, b) => (b.current?.risk_pct || b.current?.risk_score || 0) - (a.current?.risk_pct || a.current?.risk_score || 0))
    .slice(0, limit || 999);

  return `<table><thead><tr><th>Index</th><th>Event</th><th>Ridge</th><th>Dust</th><th class="num">Risk pct</th><th class="num">First</th><th class="num">lambda_q</th></tr></thead><tbody>` +
    rows.map(i => {
      const c = i.current || {};
      return `<tr>` +
        `<td><button class="pill-button" onclick="selectIndex('${esc(i.key)}')">${esc(i.key)}</button></td>` +
        `<td><span class="${stateClass(c.state)}">${esc(c.state)}</span></td>` +
        `<td><span class="${stateClass(c.ridge_state)}">${esc(c.ridge_state || 'NA')}</span></td>` +
        `<td><span class="${stateClass(c.dust_state)}">${esc(c.dust_state || 'NA')}</span></td>` +
        `<td class="num bar-cell">${num(c.risk_pct, 0)}${bar(c.risk_pct ?? c.risk_score)}</td>` +
        `<td class="num">${num(c.first_notice_score, 0)}</td>` +
        `<td class="num">${num(c.lambda_q, 1)}</td>` +
        `</tr>`;
    }).join('') +
    `</tbody></table>`;
}

window.selectIndex = function(key) {
  activeIndex = key;
  $('indexSelect').value = key;
  renderAll();
};

function leaderTable(kind) {
  const field = kind === 'bull' ? 'bull_score' : 'risk_pct';

  const rows = [...(bundle.indices || [])]
    .sort((a, b) => (b.current?.[field] || 0) - (a.current?.[field] || 0))
    .slice(0, 10);

  return `<table><thead><tr><th>Index</th><th>Name</th><th class="num">${kind}</th><th class="num">lambda flicker</th></tr></thead><tbody>` +
    rows.map(i => `<tr><td>${esc(i.key)}</td><td>${esc(i.name)}</td><td class="num">${num(i.current?.[field], 1)}</td><td class="num">${num(i.current?.lambda_flicker_score, 1)}</td></tr>`).join('') +
    `</tbody></table>`;
}

function backtestTable(indexKey) {
  const target = $('targetSelect').value;
  let rows = (bundle.backtests || []).filter(r => r.target === target);

  if (indexKey) rows = rows.filter(r => r.index === indexKey);

  rows = rows.sort((a, b) => (a.index > b.index ? 1 : -1) || (a.model > b.model ? 1 : -1));

  return `<table><thead><tr><th>Index</th><th>Model</th><th class="num">AUC</th><th class="num">Prec.</th><th class="num">Recall</th><th class="num">False</th><th class="num">Events</th></tr></thead><tbody>` +
    rows.map(r => `<tr><td>${esc(r.index)}</td><td>${esc(r.model)}</td><td class="num">${num(r.auc, 3)}</td><td class="num">${pct(r.precision, 1)}</td><td class="num">${pct(r.recall, 1)}</td><td class="num">${pct(r.false_alert_rate, 1)}</td><td class="num">${esc(r.events)}</td></tr>`).join('') +
    `</tbody></table>`;
}

function caseTable(indexKey, limit) {
  const kind = isBullTarget() ? 'bull' : 'crash';
  let rows = (bundle.case_studies || []).filter(r => r.kind === kind);

  if (indexKey) rows = rows.filter(r => r.index === indexKey);

  rows = rows
    .sort((a, b) => (a.window > b.window ? 1 : -1) || (a.index > b.index ? 1 : -1))
    .slice(0, limit || 999);

  return `<table><thead><tr><th>Window</th><th>Index</th><th class="num">Move</th><th class="num">Pre</th><th class="num">During</th><th>First warning</th><th class="num">Lead</th></tr></thead><tbody>` +
    rows.map(r => `<tr><td>${esc(r.window)}</td><td>${esc(r.index)}</td><td class="num ${Number(r.realized_move) < 0 ? 'bad' : 'good'}">${pct(r.realized_move, 1)}</td><td class="num">${num(r.max_pre_score, 1)}</td><td class="num">${num(r.max_during_score, 1)}</td><td>${esc(r.first_warning || '-')}</td><td class="num">${r.lead_days == null ? '-' : `${esc(r.lead_days)}d`}</td></tr>`).join('') +
    `</tbody></table>`;
}

function lambdaTable(limit) {
  const rows = (idx()?.s2_tail || []).slice(-limit).reverse();

  return `<table><thead><tr><th>Date</th><th class="num">lambda_q</th><th class="num">Beta</th><th class="num">R2</th><th class="num">Delta BIC</th><th class="num">Flicker</th></tr></thead><tbody>` +
    rows.map(r => `<tr><td>${esc(r.date)}</td><td class="num">${num(r.lambda_q, 2)}</td><td class="num">${num(r.beta, 3)}</td><td class="num">${num(r.r2, 3)}</td><td class="num">${num(r.delta_bic_vs_d1, 1)}</td><td class="num">${num(r.lambda_flicker_score, 1)}</td></tr>`).join('') +
    `</tbody></table>`;
}

function dustRankTable() {
  const rows = [...(bundle.indices || [])]
    .sort((a, b) => (b.current?.first_notice_score || 0) - (a.current?.first_notice_score || 0));

  return `<table><thead><tr><th>Index</th><th>Dust</th><th class="num">First</th><th class="num">Dust z</th><th class="num">Accel</th><th class="num">Risk pct</th></tr></thead><tbody>` +
    rows.map(i => `<tr><td>${esc(i.key)}</td><td><span class="${stateClass(i.current?.dust_state)}">${esc(i.current?.dust_state || 'NA')}</span></td><td class="num">${num(i.current?.first_notice_score, 1)}</td><td class="num">${num(i.current?.dust_z, 2)}</td><td class="num">${num(i.current?.dust_accel_z, 2)}</td><td class="num">${num(i.current?.risk_pct, 0)}</td></tr>`).join('') +
    `</tbody></table>`;
}

function healthTable() {
  return `<table><thead><tr><th>Index</th><th>Status</th><th>Source</th><th>Message</th></tr></thead><tbody>` +
    (bundle.health || []).map(r => `<tr><td>${esc(r.index)}</td><td>${esc(r.status)}</td><td>${esc(r.source)}</td><td>${esc(r.message)}</td></tr>`).join('') +
    `</tbody></table>`;
}

function overviewStory() {
  const s = bundle.summary || {};
  const i = idx();
  const c = i?.current || {};

  const modeLine = isBullTarget()
    ? 'Bull mode shows bull inertia, raw bull, and crash-gate suppression. It is not a safety label.'
    : 'Crash mode separates event risk, ridge failure, and first-notice dust pressure. High dust can now mark WATCH even when the ridge still holds.';

  return `<div class="small-story">` +
    `<div class="callout">${esc(i?.key || '')}: ${esc(c.state || 'NA')} - event ${esc(c.event_state || 'NA')}, ridge ${esc(c.ridge_state || 'NA')}, dust ${esc(c.dust_state || 'NA')}.</div>` +
    `<div>${esc(modeLine)}</div>` +
    `<div>The useful question is not next-bar direction. It is whether retained market structure is stable, thickening, or breaking.</div>` +
    `<div>Single-screen rule: green/yellow/red, chart, selected read, matrix, scorecard, and event lab are all visible without page scroll on desktop.</div>` +
    `<div>Global read: ${esc(s.global_state || 'NA')}; median first-notice ${num(s.median_first_notice, 1)}, median ridge-failure ${num(s.median_ridge_failure, 1)}, median lambda flicker ${num(s.median_lambda_flicker, 1)}.</div>` +
    `</div>`;
}

function methodNote() {
  return `<ul class="audit-list">` +
    `<li>Ridge = low-frequency retained market structure.</li>` +
    `<li>Dust cloud = operational residual around the ridge, not revived S1.</li>` +
    `<li>First-notice score = dust thickening and residual pressure before confirmed ridge failure.</li>` +
    `<li>Ridge-failure score = pullback failure, drawdown, ridge flattening, and lambda_q instability.</li>` +
    `<li>Risk percentile = non-leaky trailing percentile, replacing dead fixed thresholds.</li>` +
    `<li>Crash score view does not plot bull score as safety, because old ridge inertia can remain high during active crashes.</li>` +
    `</ul>`;
}

function caveats() {
  const c = bundle.metadata?.caveats || [];
  return `<ul class="audit-list">${c.map(x => `<li>${esc(x)}</li>`).join('') || '<li>No caveats in bundle.</li>'}</ul>`;
}

window.addEventListener('resize', () => {
  if (chart) chart.resize();
});

boot();
