let allRegionsData = [];
let currentFilter = 'ALL';
let countdownSeconds = 60;

document.addEventListener('DOMContentLoaded', async () => {
  await loadData();
  setupEventListeners();
  startClockAndTimers();
});

async function loadData() {
  try {
    const response = await fetch('./data/processed/risk_predictions.json?t=' + Date.now());
    if (response.ok) {
      const json = await response.json();
      allRegionsData = json.predictions || [];
    } else {
      throw new Error("Local json fetch failed");
    }
  } catch (e) {
    if (window.EMBEDDED_BIO_DATA) {
      allRegionsData = window.EMBEDDED_BIO_DATA;
    }
  }

  updateStats(allRegionsData);
  filterAndRender(document.getElementById('search-input')?.value || '', currentFilter);
}

function startClockAndTimers() {
  // Live UTC Clock
  setInterval(() => {
    const now = new Date();
    const clockEl = document.getElementById('utc-clock');
    if (clockEl) {
      clockEl.textContent = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
    }
  }, 1000);

  // 60-second auto-refresh countdown
  setInterval(async () => {
    countdownSeconds--;
    const counterEl = document.getElementById('refresh-counter');
    if (counterEl) {
      counterEl.textContent = `${countdownSeconds}s`;
    }

    if (countdownSeconds <= 0) {
      countdownSeconds = 60;
      await loadData();
    }
  }, 1000);
}

function updateStats(data) {
  const total = data.length;
  const high = data.filter(r => r.risk_tier === 'High').length;
  const med = data.filter(r => r.risk_tier === 'Medium').length;

  const avgTemp = total ? (data.reduce((acc, r) => acc + r.mean_temp_14d, 0) / total).toFixed(1) : 0;
  const avgHum = total ? (data.reduce((acc, r) => acc + r.mean_humidity_14d, 0) / total).toFixed(1) : 0;

  document.getElementById('stat-total').textContent = total;
  document.getElementById('stat-high').textContent = high;
  document.getElementById('stat-med').textContent = med;
  document.getElementById('stat-avg-temp').textContent = `${avgTemp}°C`;
  document.getElementById('stat-avg-hum').textContent = `${avgHum}%`;
}

function setupEventListeners() {
  const searchInput = document.getElementById('search-input');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      filterAndRender(e.target.value, currentFilter);
    });
  }

  const pills = document.querySelectorAll('.pill');
  pills.forEach(pill => {
    pill.addEventListener('click', (e) => {
      pills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      currentFilter = pill.dataset.filter;
      const searchVal = document.getElementById('search-input').value;
      filterAndRender(searchVal, currentFilter);
    });
  });
}

function filterAndRender(searchQuery, filterTier) {
  let filtered = [...allRegionsData];

  if (filterTier && filterTier !== 'ALL') {
    filtered = filtered.filter(r => r.risk_tier.toUpperCase() === filterTier.toUpperCase());
  }

  if (searchQuery) {
    const q = searchQuery.toLowerCase();
    filtered = filtered.filter(r => 
      r.name.toLowerCase().includes(q) || 
      r.country.toLowerCase().includes(q) ||
      (r.endemic_focus && r.endemic_focus.toLowerCase().includes(q))
    );
  }

  renderCards(filtered);
  renderTable(filtered);
}

function renderCards(data) {
  const container = document.getElementById('cards-grid');
  if (!container) return;

  if (data.length === 0) {
    container.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 3rem;">No matching regions found.</div>`;
    return;
  }

  container.innerHTML = data.map(r => {
    const tierClass = r.risk_tier.toLowerCase();
    const scorePct = Math.min(100, Math.max(0, r.risk_score * 100)).toFixed(0);

    return `
      <div class="region-card">
        <div>
          <div class="card-top">
            <div>
              <div class="region-name">${r.name}</div>
              <div class="region-country">${r.country} • ${r.endemic_focus || 'Vector Zone'}</div>
            </div>
            <span class="tier-badge ${tierClass}">${r.risk_tier}</span>
          </div>

          <div class="score-bar-container">
            <div class="score-header">
              <span>Risk Index</span>
              <strong>${r.risk_score.toFixed(2)}</strong>
            </div>
            <div class="score-bar-bg">
              <div class="score-bar-fill ${tierClass}" style="width: ${scorePct}%"></div>
            </div>
          </div>
        </div>

        <div class="metrics-row">
          <div>
            <div class="metric-item-val">${r.mean_temp_14d}°C</div>
            <div class="metric-item-lbl">14d Temp</div>
          </div>
          <div>
            <div class="metric-item-val">${r.mean_humidity_14d}%</div>
            <div class="metric-item-lbl">Humidity</div>
          </div>
          <div>
            <div class="metric-item-val">${r.vectorial_capacity_proxy.toFixed(2)}</div>
            <div class="metric-item-lbl">R₀ Proxy</div>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function renderTable(data) {
  const tbody = document.getElementById('table-body');
  if (!tbody) return;

  tbody.innerHTML = data.map(r => `
    <tr>
      <td><strong>${r.name}</strong></td>
      <td>${r.country}</td>
      <td>${r.endemic_focus || 'N/A'}</td>
      <td>${r.mean_temp_14d}°C</td>
      <td>${r.mean_humidity_14d}%</td>
      <td>${r.total_precip_14d} mm</td>
      <td><code>${r.vectorial_capacity_proxy.toFixed(3)}</code></td>
      <td><strong>${r.risk_score.toFixed(2)}</strong></td>
      <td><span class="tier-badge ${r.risk_tier.toLowerCase()}">${r.risk_tier}</span></td>
    </tr>
  `).join('');
}
