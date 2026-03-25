(function () {
  const initialNode = document.getElementById("goalie-ladders-initial-state");
  const form = document.getElementById("goalie-ladders-form");
  const dateInput = document.getElementById("goalie-date");
  const propSelect = document.getElementById("goalie-prop");
  const teamSelect = document.getElementById("goalie-team");
  const playerSelect = document.getElementById("goalie-player");
  const sortSelect = document.getElementById("goalie-sort");
  const prevLink = document.getElementById("goalie-prev");
  const nextLink = document.getElementById("goalie-next");
  const summaryNode = document.getElementById("goalie-summary");
  const statusNode = document.getElementById("goalie-status");
  const selectedNode = document.getElementById("goalie-selected");
  const resultsNode = document.getElementById("goalie-results");

  let state = {
    payload: {},
    initialState: parseInitialState(),
    loading: false,
  };

  function parseInitialState() {
    if (!initialNode || !initialNode.textContent) {
      return {};
    }
    try {
      return JSON.parse(initialNode.textContent);
    } catch (_error) {
      return {};
    }
  }

  function applyInitialState() {
    const initial = state.initialState || {};
    if (initial.date) dateInput.value = initial.date;
    if (initial.prop) propSelect.value = initial.prop;
    if (initial.sort) sortSelect.value = initial.sort;
  }

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatPercent(value, digits) {
    if (value == null || Number.isNaN(Number(value))) return "--";
    return `${(Number(value) * 100).toFixed(digits == null ? 1 : digits)}%`;
  }

  function formatDecimal(value, digits) {
    if (value == null || Number.isNaN(Number(value))) return "--";
    return Number(value).toFixed(digits == null ? 2 : digits);
  }

  function formatSignedOdds(value) {
    if (value == null || Number.isNaN(Number(value))) return "--";
    const n = Number(value);
    return n > 0 ? `+${n}` : `${n}`;
  }

  function setSelectOptions(select, options, selected, allLabel) {
    const current = selected == null ? "" : String(selected);
    const items = Array.isArray(options) ? options : [];
    select.innerHTML = "";
    if (allLabel) {
      const emptyOption = document.createElement("option");
      emptyOption.value = "";
      emptyOption.textContent = allLabel;
      select.appendChild(emptyOption);
    }
    items.forEach((option) => {
      const node = document.createElement("option");
      node.value = String(option.value == null ? "" : option.value);
      node.textContent = String(option.label == null ? option.value : option.label);
      select.appendChild(node);
    });
    select.value = current;
    if (select.value !== current) select.value = "";
  }

  function buildQueryFromControls() {
    const params = new URLSearchParams();
    const initial = state.initialState || {};
    if (dateInput.value) params.set("date", dateInput.value);
    if (propSelect.value) params.set("prop", propSelect.value);
    if (teamSelect.value) params.set("team", teamSelect.value);
    else if (!state.payload || !Object.keys(state.payload).length) {
      if (initial.team) params.set("team", initial.team);
    }
    if (playerSelect.value) params.set("goalie", playerSelect.value);
    else if (!state.payload || !Object.keys(state.payload).length) {
      if (initial.goalie) params.set("goalie", initial.goalie);
    }
    if (sortSelect.value) params.set("sort", sortSelect.value);
    return params;
  }

  function buildPageHref(payload, targetDate) {
    const params = new URLSearchParams();
    params.set("date", targetDate || payload.date || dateInput.value || "");
    params.set("prop", payload.prop || propSelect.value || "saves");
    if (payload.selectedTeam) params.set("team", payload.selectedTeam);
    if (payload.selectedGoalie) params.set("goalie", payload.selectedGoalie);
    params.set("sort", payload.selectedSort || sortSelect.value || "team");
    return `/goalie-ladders?${params.toString()}`;
  }

  function renderSummary(payload) {
    const summary = payload.summary || {};
    summaryNode.innerHTML = `
      <div class="skater-summary-head">
        <p class="skater-label">Slate Summary</p>
        <h2>${escapeHtml(payload.propLabel || "Goalie Ladders")}</h2>
      </div>
      <div class="skater-stat-strip">
        <div class="skater-stat-chip"><span>Games</span><strong>${escapeHtml(summary.games || 0)}</strong></div>
        <div class="skater-stat-chip"><span>Visible goalies</span><strong>${escapeHtml(summary.goalies || 0)}</strong></div>
        <div class="skater-stat-chip"><span>Teams</span><strong>${escapeHtml(summary.availableTeams || 0)}</strong></div>
        <div class="skater-stat-chip"><span>Sim count</span><strong>${escapeHtml((summary.simCounts || []).join(", ") || payload.defaultSims || "--")}</strong></div>
      </div>
    `;
  }

  function renderStatus(payload) {
    const selectedBits = [];
    if (payload.selectedTeam) selectedBits.push(`Team ${escapeHtml(payload.selectedTeam)}`);
    if (payload.selectedGoalie) {
      const option = (payload.goalieOptions || []).find((item) => String(item.value) === String(payload.selectedGoalie));
      selectedBits.push(option ? escapeHtml(option.goalieName || option.label) : "Goalie filter active");
    }
    const filterText = selectedBits.length ? selectedBits.join(" · ") : "All goalies visible";
    const errorText = payload.error ? "No rows matched the current filter set." : "Exact save distributions are shown from the boxscore simulation histogram.";
    statusNode.innerHTML = `
      <div class="skater-status-head">
        <p class="skater-label">Context</p>
        <h2>${escapeHtml(payload.date || "")}</h2>
      </div>
      <p class="skater-status-line">${filterText}</p>
      <p class="skater-status-line">${escapeHtml(errorText)}</p>
      <p class="skater-status-meta">Source: ${escapeHtml(payload.sourceDir || "Unavailable")}</p>
      <p class="skater-status-meta">Market lines: ${escapeHtml(payload.marketSource || "Unavailable")}</p>
    `;
  }

  function renderMarketChips(row) {
    const chips = Array.isArray(row.marketLinesByStat) ? row.marketLinesByStat : [];
    if (!chips.length) return '<div class="skater-market-chip skater-market-chip-muted">No market lines</div>';
    return chips.map((item) => {
      const lineText = item.line == null ? "--" : formatDecimal(item.line, item.line % 1 === 0 ? 0 : 1);
      return `
        <div class="skater-market-chip${item.stat === state.payload.prop ? " skater-market-chip-active" : ""}">
          <span>${escapeHtml(item.label || item.stat || "Prop")}</span>
          <strong>${escapeHtml(lineText)}</strong>
          <small>O ${escapeHtml(formatSignedOdds(item.overOdds))} / U ${escapeHtml(formatSignedOdds(item.underOdds))}</small>
        </div>
      `;
    }).join("");
  }

  function renderLadderRows(row) {
    const ladder = Array.isArray(row.ladder) ? row.ladder : [];
    if (!ladder.length) return '<div class="skater-empty-card">No distribution rows available.</div>';
    return ladder.map((entry) => {
      const exactWidth = Math.max(2, Math.min(100, Number(entry.exactProb || 0) * 100));
      const hitWidth = Math.max(2, Math.min(100, Number(entry.hitProb || 0) * 100));
      return `
        <div class="skater-ladder-row">
          <div class="skater-ladder-total">${escapeHtml(entry.total)}</div>
          <div class="skater-ladder-metrics">
            <div class="skater-ladder-bar-shell">
              <span>Exact</span>
              <div class="skater-ladder-bar"><i style="width:${exactWidth}%"></i></div>
              <strong>${escapeHtml(entry.exactCount)}</strong>
              <em>${escapeHtml(formatPercent(entry.exactProb, 1))}</em>
            </div>
            <div class="skater-ladder-bar-shell skater-ladder-bar-shell-hit">
              <span>At least</span>
              <div class="skater-ladder-bar"><i style="width:${hitWidth}%"></i></div>
              <strong>${escapeHtml(entry.hitCount)}</strong>
              <em>${escapeHtml(formatPercent(entry.hitProb, 1))}</em>
            </div>
          </div>
        </div>
      `;
    }).join("");
  }

  function renderSelected(payload) {
    const rows = Array.isArray(payload.rows) ? payload.rows : [];
    const activeRow = payload.selectedGoalie ? rows[0] : null;
    if (!activeRow) {
      selectedNode.hidden = true;
      selectedNode.innerHTML = "";
      return;
    }
    selectedNode.hidden = false;
    selectedNode.innerHTML = `
      <article class="skater-selected-card">
        <div class="skater-selected-top">
          <div class="skater-selected-player">
            <img class="skater-headshot" src="${escapeHtml(activeRow.headshotUrl || "")}" alt="${escapeHtml(activeRow.goalieName || "Goalie")}" loading="lazy" />
            <div>
              <p class="skater-label">Selected goalie</p>
              <h2>${escapeHtml(activeRow.goalieName || "")}</h2>
              <p>${escapeHtml(activeRow.matchup || "")}</p>
            </div>
          </div>
          <div class="skater-selected-logos">
            ${activeRow.teamLogoUrl ? `<img src="${escapeHtml(activeRow.teamLogoUrl)}" alt="${escapeHtml(activeRow.team || "Team")}" loading="lazy" />` : ""}
            ${activeRow.opponentLogoUrl ? `<img src="${escapeHtml(activeRow.opponentLogoUrl)}" alt="${escapeHtml(activeRow.opponent || "Opponent")}" loading="lazy" />` : ""}
          </div>
        </div>
        <div class="skater-selected-stats">
          <div><span>Mean</span><strong>${escapeHtml(formatDecimal(activeRow.mean, 2))}</strong></div>
          <div><span>Mode</span><strong>${escapeHtml(activeRow.mode)}</strong></div>
          <div><span>Mode prob</span><strong>${escapeHtml(formatPercent(activeRow.modeProb, 1))}</strong></div>
          <div><span>Line</span><strong>${escapeHtml(activeRow.marketLine == null ? "--" : formatDecimal(activeRow.marketLine, activeRow.marketLine % 1 === 0 ? 0 : 1))}</strong></div>
          <div><span>Over line</span><strong>${escapeHtml(formatPercent(activeRow.overLineProb, 1))}</strong></div>
        </div>
      </article>
    `;
  }

  function renderResults(payload) {
    const rows = Array.isArray(payload.rows) ? payload.rows : [];
    if (!rows.length) {
      resultsNode.innerHTML = '<div class="skater-empty-card">No goalies available for this date and filter set.</div>';
      return;
    }
    resultsNode.innerHTML = rows.map((row) => {
      const lineText = row.marketLine == null ? "--" : formatDecimal(row.marketLine, Number(row.marketLine) % 1 === 0 ? 0 : 1);
      return `
        <article class="skater-card">
          <header class="skater-card-head">
            <div class="skater-card-player">
              <img class="skater-headshot" src="${escapeHtml(row.headshotUrl || "")}" alt="${escapeHtml(row.goalieName || "Goalie")}" loading="lazy" />
              <div>
                <p class="skater-label">${escapeHtml(row.team || "")}${row.opponent ? ` vs ${escapeHtml(row.opponent)}` : ""}</p>
                <h3>${escapeHtml(row.goalieName || "")}</h3>
                <p>${escapeHtml(row.matchup || "")}</p>
              </div>
            </div>
            <div class="skater-card-logos">
              ${row.teamLogoUrl ? `<img src="${escapeHtml(row.teamLogoUrl)}" alt="${escapeHtml(row.team || "Team")}" loading="lazy" />` : ""}
              ${row.opponentLogoUrl ? `<img src="${escapeHtml(row.opponentLogoUrl)}" alt="${escapeHtml(row.opponent || "Opponent")}" loading="lazy" />` : ""}
            </div>
          </header>

          <div class="skater-card-stats">
            <div><span>Mean</span><strong>${escapeHtml(formatDecimal(row.mean, 2))}</strong></div>
            <div><span>Mode</span><strong>${escapeHtml(row.mode)}</strong></div>
            <div><span>Mode prob</span><strong>${escapeHtml(formatPercent(row.modeProb, 1))}</strong></div>
            <div><span>Line</span><strong>${escapeHtml(lineText)}</strong></div>
            <div><span>Over line</span><strong>${escapeHtml(formatPercent(row.overLineProb, 1))}</strong></div>
            <div><span>Sims</span><strong>${escapeHtml(row.simCount)}</strong></div>
          </div>

          <div class="skater-market-row">${renderMarketChips(row)}</div>
          <div class="skater-ladder-table">${renderLadderRows(row)}</div>
        </article>
      `;
    }).join("");
  }

  function applyPayload(payload) {
    state.payload = payload || {};
    dateInput.value = payload.date || dateInput.value || "";
    propSelect.value = payload.prop || "saves";
    sortSelect.value = payload.selectedSort || "team";
    setSelectOptions(teamSelect, payload.teamOptions || [], payload.selectedTeam || "", "All teams");
    setSelectOptions(playerSelect, payload.goalieOptions || [], payload.selectedGoalie || "", "All goalies");
    const nav = payload.nav || {};
    if (nav.prevDate) {
      prevLink.href = buildPageHref(payload, nav.prevDate);
      prevLink.dataset.date = nav.prevDate;
      prevLink.classList.remove("is-disabled");
    } else {
      prevLink.href = "#";
      prevLink.dataset.date = "";
      prevLink.classList.add("is-disabled");
    }
    if (nav.nextDate) {
      nextLink.href = buildPageHref(payload, nav.nextDate);
      nextLink.dataset.date = nav.nextDate;
      nextLink.classList.remove("is-disabled");
    } else {
      nextLink.href = "#";
      nextLink.dataset.date = "";
      nextLink.classList.add("is-disabled");
    }
    renderSummary(payload);
    renderStatus(payload);
    renderSelected(payload);
    renderResults(payload);
  }

  async function refresh() {
    if (state.loading) return;
    state.loading = true;
    resultsNode.innerHTML = '<div class="skater-empty-card">Loading goalie ladders…</div>';
    try {
      const params = buildQueryFromControls();
      const response = await fetch(`/api/goalie-ladders?${params.toString()}`, { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      applyPayload(payload);
      window.history.replaceState({}, "", `/goalie-ladders?${params.toString()}`);
    } catch (error) {
      statusNode.innerHTML = `<div class="skater-status-error">Unable to load goalie ladders: ${escapeHtml(error && error.message ? error.message : "Unknown error")}</div>`;
      resultsNode.innerHTML = '<div class="skater-empty-card">The goalie ladders request failed.</div>';
    } finally {
      state.loading = false;
    }
  }

  function handleControlChange(event) {
    if (event.target === teamSelect) playerSelect.value = "";
    refresh();
  }

  form.addEventListener("submit", function (event) { event.preventDefault(); refresh(); });
  [dateInput, propSelect, teamSelect, playerSelect, sortSelect].forEach((node) => node.addEventListener("change", handleControlChange));
  [prevLink, nextLink].forEach((node) => {
    node.addEventListener("click", function (event) {
      const targetDate = node.dataset.date || "";
      if (!targetDate) { event.preventDefault(); return; }
      event.preventDefault();
      dateInput.value = targetDate;
      refresh();
    });
  });

  applyInitialState();
  refresh();
})();