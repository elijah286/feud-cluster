/* ================================================================
   LabVIEW Family Feud — frontend logic
   ================================================================ */

// Global state
let appState = {
  currentRun: null,       // { source_file, created_at, top_k, prompts, prompt_column_order, ... }
  dirty: false,
  selectedAnswers: [],    // [{ promptIdx, clusterIdx, text }]
  selectedFile: null,     // File object for pending upload
  mergeState: null,       // { promptIdx, clusterIndices: [] }
  splitState: null,       // { promptIdx, clusterIdx }
  dragState: null,        // { promptIdx, clusterIdx, text }
  savedAs: null,          // filename of the current run in the DB
  lastKnownVersion: null, // updated_at timestamp for change detection
  justSaved: false,       // flag to skip refetch after own save
};

// ── Helpers ──────────────────────────────────────────────────────

function toast(msg, type = "info") {
  const el = document.getElementById("toast");
  document.getElementById("toastMsg").textContent = msg;
  el.className = `toast ${type} visible`;
  clearTimeout(el._timer);
  // Errors stay until dismissed; info/success auto-hide after 3.5s
  if (type !== "error") {
    el._timer = setTimeout(() => el.classList.remove("visible"), 3500);
  }
}

function dismissToast() {
  const el = document.getElementById("toast");
  clearTimeout(el._timer);
  el.classList.remove("visible");
}

function showLoading(msg = "Loading...") {
  document.getElementById("loadingMsg").textContent = msg;
  document.getElementById("loadingOverlay").classList.remove("hidden");
}
function hideLoading() {
  document.getElementById("loadingOverlay").classList.add("hidden");
}

let _autoSaveTimer = null;

function markDirty() {
  appState.dirty = true;
  const el = document.getElementById("saveStatus");
  el.textContent = "Unsaved";
  el.className = "save-status unsaved";
  el.classList.remove("hidden");
  // Debounced auto-save: waits 1.5s after last change
  clearTimeout(_autoSaveTimer);
  _autoSaveTimer = setTimeout(() => autoSave(), 1500);
}

function markSaved() {
  appState.dirty = false;
  const el = document.getElementById("saveStatus");
  el.textContent = "\u2713 Saved";
  el.className = "save-status saved";
  el.classList.remove("hidden");
}

function clusterTotal(cluster) {
  return (cluster.members || []).reduce((s, m) => s + (m.count || 1), 0);
}

function sortClustersByCount(clusters) {
  return [...clusters].sort((a, b) => clusterTotal(b) - clusterTotal(a));
}

function clearDropTargets() {
  document.querySelectorAll(".cluster-card.drop-target").forEach(el => el.classList.remove("drop-target"));
}

function clearDraggedTag() {
  document.querySelectorAll(".answer-tag.dragging").forEach(el => el.classList.remove("dragging"));
}

// ── Navigation ───────────────────────────────────────────────────

function showPage(page) {
  document.getElementById("pageLanding").classList.toggle("hidden", page !== "landing");
  document.getElementById("pageReview").classList.toggle("hidden", page !== "review");
  document.getElementById("btnExport").classList.toggle("hidden", page !== "review");
  document.getElementById("btnBack").classList.toggle("hidden", page !== "review");
  document.getElementById("btnUploadMore").classList.toggle("hidden", page !== "review");
  const statusEl = document.getElementById("saveStatus");
  if (page === "review") {
    statusEl.textContent = "\u2713 Saved";
    statusEl.className = "save-status saved";
    statusEl.classList.remove("hidden");
  } else {
    document.getElementById("moveBar").classList.remove("visible");
    statusEl.classList.add("hidden");
  }
}

function showUploadPanel() {
  // Switch to landing but keep the current run for seed clusters
  showPage("landing");
  document.getElementById("uploadConfig").classList.add("hidden");
  document.getElementById("seedOption").classList.remove("hidden");
}

function goHome() {
  // Auto-save flushes on dirty, so just save immediately if needed
  if (appState.dirty && appState.currentRun) {
    clearTimeout(_autoSaveTimer);
    autoSave();
  }
  appState.currentRun = null;
  appState.dirty = false;
  appState.selectedAnswers = [];
  showPage("landing");
  loadRunsList();
}

// ── Landing page ─────────────────────────────────────────────────

async function loadLatestRun() {
  showLoading("Loading latest session...");
  try {
    const resp = await fetch("/api/runs/latest");
    if (resp.ok) {
      appState.currentRun = await resp.json();
      appState.savedAs = appState.currentRun.saved_as || null;
      appState.dirty = false;
      renderReview();
      showPage("review");
      hideLoading();
      fetchVersion();
      startSync();
      return;
    }
  } catch (e) {
    console.error("Failed to load latest run:", e);
  }
  // No run found — show landing page
  hideLoading();
  showPage("landing");
  loadRunsList();
}

async function loadRunsList() {
  try {
    const resp = await fetch("/api/runs");
    const runs = await resp.json();
    const container = document.getElementById("runsList");
    const card = document.getElementById("openRunCard");

    if (runs.length === 0) {
      container.classList.add("hidden");
      card.querySelector("p").textContent = "No saved runs yet";
      card.onclick = null;
      card.style.cursor = "default";
      card.style.opacity = "0.5";
      return;
    }

    card.onclick = () => {
      container.classList.toggle("hidden");
    };
    card.style.cursor = "pointer";
    card.style.opacity = "1";

    container.innerHTML = runs.map(r => `
      <div class="run-item" onclick="openRun('${r.filename}')">
        <div class="run-item-info">
          <div class="run-item-name">${escHtml(r.source_file || r.filename)}</div>
          <div class="run-item-meta">${r.n_prompts} prompt(s) · ${r.created_at ? new Date(r.created_at).toLocaleString() : "unknown date"} · ${escHtml(r.filename)}</div>
        </div>
        <span style="color: var(--primary); font-size: 1.1rem;">→</span>
      </div>
    `).join("");
    container.classList.remove("hidden");
  } catch (e) {
    console.error("Failed to load runs:", e);
  }
}

async function openRun(filename) {
  showLoading("Loading saved run...");
  try {
    const resp = await fetch(`/api/runs/${encodeURIComponent(filename)}`);
    if (!resp.ok) throw new Error(await resp.text());
    appState.currentRun = await resp.json();
    appState.savedAs = filename;
    appState.dirty = false;
    renderReview();
    showPage("review");
    fetchVersion();
    startSync();
    toast("Run loaded", "success");
  } catch (e) {
    toast("Failed to load run: " + e.message, "error");
  }
  hideLoading();
}

// ── Upload & Cluster ─────────────────────────────────────────────

function handleUpload(input) {
  const file = input.files[0];
  if (!file) return;
  appState.selectedFile = file;
  document.getElementById("uploadFileName").textContent = `File: ${file.name}`;
  document.getElementById("uploadConfig").classList.remove("hidden");
  // Reset so re-selecting the same file triggers onchange again
  input.value = "";

  // If there's an existing run with cluster labels, offer seeding
  if (appState.currentRun) {
    document.getElementById("seedOption").classList.remove("hidden");
  }
}

async function runClustering() {
  const file = appState.selectedFile;
  if (!file) { toast("No file selected", "error"); return; }

  const topK = document.getElementById("topKInput").value || "10";
  const ignore = document.getElementById("ignoreInput").value || "";

  const form = new FormData();
  form.append("file", file);
  form.append("top_k", topK);
  form.append("ignore_exact", ignore);

  // Build seed clusters from prior run if checkbox is checked
  if (appState.currentRun && document.getElementById("useSeedCheck")?.checked) {
    const seeds = {};
    const prompts = appState.currentRun.prompts || {};
    for (const [col, pdata] of Object.entries(prompts)) {
      seeds[col] = (pdata.clusters || []).map(c => c.label);
    }
    form.append("seed_clusters", JSON.stringify(seeds));
  }

  showLoading("Running LLM clustering on all prompts... This may take a minute.");
  try {
    const resp = await fetch("/api/upload-and-cluster", { method: "POST", body: form });
    if (!resp.ok) {
      let errMsg = "Clustering failed";
      try {
        const err = await resp.json();
        errMsg = err.error || (err.details ? err.details.join("; ") : errMsg);
      } catch {
        const text = await resp.text().catch(() => "");
        errMsg = `Server error (${resp.status})${text ? ": " + text.slice(0, 200) : ""}`;
      }
      throw new Error(errMsg);
    }
    appState.currentRun = await resp.json();
    appState.savedAs = appState.currentRun.saved_as || null;
    appState.dirty = false;
    document.getElementById("uploadConfig").classList.add("hidden");
    renderReview();
    showPage("review");
    toast(`Clustered ${Object.keys(appState.currentRun.prompts).length} prompt(s).`, "success");
    // Fetch version so polling knows the baseline
    fetchVersion();
    startSync();
  } catch (e) {
    toast("Clustering failed: " + e.message, "error");
  }
  hideLoading();
}

// ── Save ─────────────────────────────────────────────────────────

let _saving = false;

async function autoSave() {
  if (!appState.currentRun || _saving) return;
  _saving = true;
  const el = document.getElementById("saveStatus");
  el.textContent = "Saving…";
  el.className = "save-status saving";
  el.classList.remove("hidden");
  try {
    const payload = { ...appState.currentRun };
    if (appState.savedAs) payload.saved_as = appState.savedAs;
    const resp = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const result = await resp.json();
    if (result.saved_as) appState.savedAs = result.saved_as;
    appState.justSaved = true;
    markSaved();
    fetchVersion();
  } catch (e) {
    el.textContent = "Save failed";
    el.className = "save-status unsaved";
  }
  _saving = false;
}

// ── Export ────────────────────────────────────────────────────────

async function exportFeud() {
  if (!appState.currentRun) return;

  try {
    const resp = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(appState.currentRun),
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `feud_export_${slugify(appState.currentRun.source_file || "run")}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast("Exported!", "success");
  } catch (e) {
    toast("Export failed: " + e.message, "error");
  }
}

function slugify(s) {
  return s.toLowerCase().replace(/[^\w]+/g, "_").replace(/^_|_$/g, "").slice(0, 40);
}

// ── Render Review Page ───────────────────────────────────────────

function renderReview() {
  const container = document.getElementById("pageReview");
  const run = appState.currentRun;
  if (!run || !run.prompts) { container.innerHTML = "<p>No data.</p>"; return; }

  const order = run.prompt_column_order || Object.keys(run.prompts);
  let html = "";

  order.forEach((col, promptIdx) => {
    const pdata = run.prompts[col];
    if (!pdata) return;
    const clusters = pdata.clusters || [];
    const totalResponses = clusters.reduce((s, c) => s + clusterTotal(c), 0);
    const limit = pdata.artifact?.top_k || 10;

    // Keep clusters in current order (user controls sorting explicitly)
    const sorted = clusters.map((c, i) => ({ ...c, _origIdx: i }));

    html += `
      <div class="prompt-section" data-prompt="${promptIdx}" data-col="${escAttr(col)}">
        <div class="prompt-header" onclick="togglePrompt(${promptIdx})">
          <div class="prompt-title">${escHtml(col)}</div>
          <div class="prompt-stats">
            <span>${clusters.length} clusters</span>
            <span>${totalResponses} responses</span>
          </div>
          <span class="prompt-toggle">▼</span>
        </div>
        <div class="prompt-body">
          <div class="prompt-toolbar">
            <button class="btn btn-sm" onclick="createCluster(${promptIdx})">+ New Cluster</button>
            <button class="btn btn-sm" onclick="sortClusters(${promptIdx})">Sort by responses ↓</button>
            <button class="btn btn-sm" onclick="deleteEmptyClusters(${promptIdx})">Remove Empty</button>
            <button class="btn btn-sm" onclick="startMergeMode(${promptIdx})">Merge…</button>
          </div>
          <div class="cluster-grid" id="clusterGrid_${promptIdx}">
            ${sorted.map((c, displayIdx) => renderClusterCard(c, promptIdx, c._origIdx)).join("")}
          </div>
        </div>
      </div>`;
  });

  container.innerHTML = html;
}

function renderClusterCard(cluster, promptIdx, clusterIdx) {
  const total = clusterTotal(cluster);
  const members = [...(cluster.members || [])].sort((a, b) => (b.count || 1) - (a.count || 1));
  const excluded = !!cluster.excluded;

  return `
    <div class="cluster-card ${excluded ? "excluded" : ""}" data-prompt="${promptIdx}" data-cluster="${clusterIdx}"
         id="card_${promptIdx}_${clusterIdx}">
      <div class="cluster-card-header">
        <span class="cluster-label" contenteditable="true"
              onblur="renameCluster(${promptIdx}, ${clusterIdx}, this.textContent)"
              onkeydown="if(event.key==='Enter'){event.preventDefault();this.blur();}"
              >${escHtml(cluster.label)}</span>
        <span class="cluster-count">${total}</span>
        <div class="cluster-card-actions">
          <button class="btn btn-sm" onclick="startSplit(${promptIdx}, ${clusterIdx})" title="Split this cluster">Split</button>
          <button class="btn btn-sm ${excluded ? 'btn-primary' : ''}" onclick="toggleExclude(${promptIdx}, ${clusterIdx})" title="${excluded ? 'Include in export' : 'Exclude from export'}">${excluded ? 'Include' : 'Exclude'}</button>
          <button class="btn btn-sm btn-danger" onclick="deleteCluster(${promptIdx}, ${clusterIdx})" title="Delete cluster">✕</button>
        </div>
      </div>
      <div class="cluster-divider"></div>
      <div class="answer-list">
        ${members.map((m, mi) => `
          <span class="answer-tag ${isSelected(promptIdx, clusterIdx, m.text) ? "checked" : ""}"
                draggable="true"
                data-prompt="${promptIdx}" data-cluster="${clusterIdx}" data-member="${mi}">
            ${escHtml(m.text)} <span class="count">×${m.count}</span>
          </span>
        `).join("")}
      </div>
    </div>`;
}

// ── Prompt collapse ──────────────────────────────────────────────

function togglePrompt(promptIdx) {
  const section = document.querySelector(`[data-prompt="${promptIdx}"].prompt-section`);
  if (section) section.classList.toggle("collapsed");
}

// ── Exclude toggle ───────────────────────────────────────────────

function toggleExclude(promptIdx, clusterIdx) {
  const run = appState.currentRun;
  const order = run.prompt_column_order || Object.keys(run.prompts);
  const col = order[promptIdx];
  const cluster = run.prompts[col].clusters[clusterIdx];
  cluster.excluded = !cluster.excluded;
  markDirty();
  renderReview();
  toast(cluster.excluded ? `"${cluster.label}" excluded from export` : `"${cluster.label}" included in export`, "info");
}

// ── Answer selection ─────────────────────────────────────────────

function isSelected(promptIdx, clusterIdx, text) {
  return appState.selectedAnswers.some(
    a => a.promptIdx === promptIdx && a.clusterIdx === clusterIdx && a.text === text
  );
}

function getAnswerText(promptIdx, clusterIdx, memberIdx) {
  const run = appState.currentRun;
  const order = run.prompt_column_order || Object.keys(run.prompts);
  const col = order[promptIdx];
  const cluster = run.prompts[col].clusters[clusterIdx];
  if (!cluster) return null;
  const members = [...(cluster.members || [])].sort((a, b) => (b.count || 1) - (a.count || 1));
  return members[memberIdx]?.text || null;
}

function toggleAnswer(promptIdx, clusterIdx, text, el) {
  const idx = appState.selectedAnswers.findIndex(
    a => a.promptIdx === promptIdx && a.clusterIdx === clusterIdx && a.text === text
  );
  if (idx >= 0) {
    appState.selectedAnswers.splice(idx, 1);
    el.classList.remove("checked");
  } else {
    // Only allow selecting from the same prompt
    if (appState.selectedAnswers.length > 0 && appState.selectedAnswers[0].promptIdx !== promptIdx) {
      toast("Select answers from one prompt at a time", "info");
      return;
    }
    appState.selectedAnswers.push({ promptIdx, clusterIdx, text });
    el.classList.add("checked");
  }
  updateMoveBar();
}

function clearSelection() {
  appState.selectedAnswers = [];
  document.querySelectorAll(".answer-tag.checked").forEach(el => el.classList.remove("checked"));
  updateMoveBar();
}

function updateMoveBar() {
  const bar = document.getElementById("moveBar");
  const count = appState.selectedAnswers.length;
  document.getElementById("moveCount").textContent = count;

  if (count === 0) {
    bar.classList.remove("visible");
    return;
  }

  bar.classList.add("visible");

  // Populate destination dropdown with clusters from the same prompt
  const promptIdx = appState.selectedAnswers[0].promptIdx;
  const run = appState.currentRun;
  const order = run.prompt_column_order || Object.keys(run.prompts);
  const col = order[promptIdx];
  const clusters = run.prompts[col].clusters || [];

  const sourceClusterIdxs = new Set(appState.selectedAnswers.map(a => a.clusterIdx));
  const select = document.getElementById("moveDest");
  select.innerHTML = clusters.map((c, i) => {
    const isSrc = sourceClusterIdxs.has(i);
    return `<option value="${i}" ${isSrc ? "disabled" : ""}>${c.label} (${clusterTotal(c)})${isSrc ? " — source" : ""}</option>`;
  }).join("");

  // Select first non-source
  for (let i = 0; i < clusters.length; i++) {
    if (!sourceClusterIdxs.has(i)) {
      select.value = i;
      break;
    }
  }
}

function moveSelected() {
  if (appState.selectedAnswers.length === 0) return;

  const destIdx = parseInt(document.getElementById("moveDest").value);
  const promptIdx = appState.selectedAnswers[0].promptIdx;
  const run = appState.currentRun;
  const order = run.prompt_column_order || Object.keys(run.prompts);
  const col = order[promptIdx];
  const clusters = run.prompts[col].clusters;

  // Move each selected answer
  for (const sel of appState.selectedAnswers) {
    const srcCluster = clusters[sel.clusterIdx];
    if (!srcCluster) continue;

    // Find and remove from source
    const memberIdx = srcCluster.members.findIndex(m => m.text === sel.text);
    if (memberIdx < 0) continue;
    const [member] = srcCluster.members.splice(memberIdx, 1);

    // Add to destination
    clusters[destIdx].members.push(member);
  }

  appState.selectedAnswers = [];
  markDirty();
  renderReview();
  toast("Answers moved", "success");
}

function moveSingleAnswer(promptIdx, srcIdx, destIdx, text) {
  const run = appState.currentRun;
  const order = run.prompt_column_order || Object.keys(run.prompts);
  const col = order[promptIdx];
  const clusters = run.prompts[col].clusters;
  const srcCluster = clusters[srcIdx];
  const destCluster = clusters[destIdx];
  if (!srcCluster || !destCluster) return;
  const memberIdx = srcCluster.members.findIndex(m => m.text === text);
  if (memberIdx < 0) return;
  const [member] = srcCluster.members.splice(memberIdx, 1);
  destCluster.members.push(member);
  appState.selectedAnswers = [];
  markDirty();
  renderReview();
  toast("Answer moved", "success");
}

function sortClusters(promptIdx) {
  const run = appState.currentRun;
  const order = run.prompt_column_order || Object.keys(run.prompts);
  const col = order[promptIdx];
  // Non-excluded sorted by count desc, then excluded sorted by count desc at the end
  run.prompts[col].clusters.sort((a, b) => {
    const aExcl = a.excluded ? 1 : 0;
    const bExcl = b.excluded ? 1 : 0;
    if (aExcl !== bExcl) return aExcl - bExcl;
    return clusterTotal(b) - clusterTotal(a);
  });
  markDirty();
  renderReview();
}

// ── Cluster operations ───────────────────────────────────────────

function renameCluster(promptIdx, clusterIdx, newLabel) {
  const run = appState.currentRun;
  const order = run.prompt_column_order || Object.keys(run.prompts);
  const col = order[promptIdx];
  const cluster = run.prompts[col].clusters[clusterIdx];
  const trimmed = newLabel.trim();
  if (!trimmed || trimmed === cluster.label) return;
  cluster.label = trimmed;
  markDirty();
}

function createCluster(promptIdx) {
  const run = appState.currentRun;
  const order = run.prompt_column_order || Object.keys(run.prompts);
  const col = order[promptIdx];
  run.prompts[col].clusters.push({ label: "New Cluster", members: [] });
  markDirty();
  renderReview();
}

function deleteCluster(promptIdx, clusterIdx) {
  const run = appState.currentRun;
  const order = run.prompt_column_order || Object.keys(run.prompts);
  const col = order[promptIdx];
  const cluster = run.prompts[col].clusters[clusterIdx];
  const total = clusterTotal(cluster);
  if (total > 0 && !confirm(`Delete "${cluster.label}" with ${total} responses? Answers will be lost.`)) return;
  run.prompts[col].clusters.splice(clusterIdx, 1);
  clearSelection();
  markDirty();
  renderReview();
}

function deleteEmptyClusters(promptIdx) {
  const run = appState.currentRun;
  const order = run.prompt_column_order || Object.keys(run.prompts);
  const col = order[promptIdx];
  const before = run.prompts[col].clusters.length;
  run.prompts[col].clusters = run.prompts[col].clusters.filter(c => (c.members || []).length > 0);
  const removed = before - run.prompts[col].clusters.length;
  if (removed > 0) {
    markDirty();
    renderReview();
    toast(`Removed ${removed} empty cluster(s)`, "info");
  } else {
    toast("No empty clusters", "info");
  }
}

// ── Merge ────────────────────────────────────────────────────────

function startMergeMode(promptIdx) {
  const run = appState.currentRun;
  const order = run.prompt_column_order || Object.keys(run.prompts);
  const col = order[promptIdx];
  const clusters = run.prompts[col].clusters || [];

  if (clusters.length < 2) {
    toast("Need at least 2 clusters to merge", "info");
    return;
  }

  // Show a modal to pick which clusters to merge
  const modal = document.getElementById("mergeModal");
  const optionsDiv = document.getElementById("mergeLabelOptions");

  optionsDiv.innerHTML = `
    <p class="text-sm mb-2">Select clusters to merge (at least 2):</p>
    ${clusters.map((c, i) => `
      <label style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px; cursor: pointer;">
        <input type="checkbox" class="merge-check" value="${i}" data-label="${escAttr(c.label)}">
        <strong>${escHtml(c.label)}</strong> <span class="text-muted text-sm">(${clusterTotal(c)} responses)</span>
      </label>
    `).join("")}
  `;
  document.getElementById("mergeCustomLabel").value = "";
  appState.mergeState = { promptIdx };
  modal.classList.remove("hidden");
}

function closeMergeModal() {
  document.getElementById("mergeModal").classList.add("hidden");
  appState.mergeState = null;
}

function confirmMerge() {
  if (!appState.mergeState) return;
  const { promptIdx } = appState.mergeState;
  const run = appState.currentRun;
  const order = run.prompt_column_order || Object.keys(run.prompts);
  const col = order[promptIdx];
  const clusters = run.prompts[col].clusters;

  const checked = [...document.querySelectorAll(".merge-check:checked")].map(el => parseInt(el.value));
  if (checked.length < 2) {
    toast("Select at least 2 clusters to merge", "info");
    return;
  }

  const customLabel = document.getElementById("mergeCustomLabel").value.trim();
  const keepLabel = customLabel || clusters[checked[0]].label;

  // Collect all members from selected clusters
  const allMembers = [];
  for (const idx of checked) {
    allMembers.push(...(clusters[idx].members || []));
  }

  // Remove merged clusters (reverse order to keep indices valid)
  const sorted = [...checked].sort((a, b) => b - a);
  for (const idx of sorted) {
    clusters.splice(idx, 1);
  }

  // Add merged cluster
  clusters.push({ label: keepLabel, members: allMembers });

  closeMergeModal();
  clearSelection();
  markDirty();
  renderReview();
  toast(`Merged ${checked.length} clusters into "${keepLabel}"`, "success");
}

// ── Split ────────────────────────────────────────────────────────

function startSplit(promptIdx, clusterIdx) {
  const run = appState.currentRun;
  const order = run.prompt_column_order || Object.keys(run.prompts);
  const col = order[promptIdx];
  const cluster = run.prompts[col].clusters[clusterIdx];

  if ((cluster.members || []).length < 2) {
    toast("Need at least 2 answers to split", "info");
    return;
  }

  const members = [...(cluster.members || [])].sort((a, b) => (b.count || 1) - (a.count || 1));
  const listDiv = document.getElementById("splitAnswerList");
  listDiv.innerHTML = members.map((m, i) => `
    <label style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px; cursor: pointer;">
      <input type="checkbox" class="split-check" data-idx="${i}">
      ${escHtml(m.text)} <span class="text-muted text-sm">×${m.count}</span>
    </label>
  `).join("");
  document.getElementById("splitNewLabel").value = "";

  appState.splitState = { promptIdx, clusterIdx };
  document.getElementById("splitModal").classList.remove("hidden");
}

function closeSplitModal() {
  document.getElementById("splitModal").classList.add("hidden");
  appState.splitState = null;
}

function confirmSplit() {
  if (!appState.splitState) return;
  const { promptIdx, clusterIdx } = appState.splitState;
  const run = appState.currentRun;
  const order = run.prompt_column_order || Object.keys(run.prompts);
  const col = order[promptIdx];
  const clusters = run.prompts[col].clusters;
  const cluster = clusters[clusterIdx];

  const checked = [...document.querySelectorAll(".split-check:checked")].map(el => parseInt(el.dataset.idx));
  if (checked.length === 0) {
    toast("Select answers to split out", "info");
    return;
  }
  if (checked.length === cluster.members.length) {
    toast("Can't move all answers — that's not a split", "info");
    return;
  }

  const newLabel = document.getElementById("splitNewLabel").value.trim() || "New Cluster";

  // Resolve indices to member texts (sorted order)
  const sortedMembers = [...(cluster.members || [])].sort((a, b) => (b.count || 1) - (a.count || 1));
  const textsToMove = new Set(checked.map(i => sortedMembers[i].text));

  const stayMembers = cluster.members.filter(m => !textsToMove.has(m.text));
  const goMembers = cluster.members.filter(m => textsToMove.has(m.text));

  cluster.members = stayMembers;
  clusters.push({ label: newLabel, members: goMembers });

  closeSplitModal();
  clearSelection();
  markDirty();
  renderReview();
  toast(`Split ${goMembers.length} answer(s) into "${newLabel}"`, "success");
}

// ── Real-time sync ───────────────────────────────────────────────

let _syncInterval = null;

async function fetchVersion() {
  try {
    const resp = await fetch("/api/runs/latest/version");
    if (resp.ok) {
      const info = await resp.json();
      appState.lastKnownVersion = info.version;
    }
  } catch { /* ignore */ }
}

function startSync() {
  stopSync();
  _syncInterval = setInterval(pollForChanges, 2000);
}

function stopSync() {
  if (_syncInterval) {
    clearInterval(_syncInterval);
    _syncInterval = null;
  }
}

async function pollForChanges() {
  if (!appState.currentRun) return;
  // Don't poll while we're in the middle of saving
  if (_saving) return;
  try {
    const resp = await fetch("/api/runs/latest/version");
    if (!resp.ok) return;
    const info = await resp.json();
    if (appState.lastKnownVersion && info.version !== appState.lastKnownVersion) {
      // Version changed
      if (appState.justSaved) {
        // It was our own save — just update the version and skip refetch
        appState.justSaved = false;
        appState.lastKnownVersion = info.version;
        return;
      }
      // Someone else changed it — refetch
      appState.lastKnownVersion = info.version;
      await refetchLatest();
    }
  } catch { /* network blip — ignore */ }
}

async function refetchLatest() {
  try {
    const resp = await fetch("/api/runs/latest");
    if (!resp.ok) return;
    const data = await resp.json();
    appState.currentRun = data;
    appState.savedAs = data.saved_as || appState.savedAs;
    appState.dirty = false;
    clearSelection();
    renderReview();
    toast("Updated — a collaborator made changes", "info");
  } catch { /* ignore */ }
}

// ── Utility ──────────────────────────────────────────────────────

function escHtml(s) {
  const d = document.createElement("div");
  d.textContent = s || "";
  return d.innerHTML;
}
function escAttr(s) {
  return (s || "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ── Init ─────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  loadLatestRun();

  // Event delegation for answer tags (avoids inline onclick with special chars)
  document.addEventListener("click", (e) => {
    const tag = e.target.closest(".answer-tag");
    if (!tag) return;
    const promptIdx = parseInt(tag.dataset.prompt);
    const clusterIdx = parseInt(tag.dataset.cluster);
    const memberIdx = parseInt(tag.dataset.member);
    const text = getAnswerText(promptIdx, clusterIdx, memberIdx);
    if (text !== null) {
      toggleAnswer(promptIdx, clusterIdx, text, tag);
    }
  });

  // ── Drag-and-drop ──
  document.addEventListener("dragstart", (e) => {
    const tag = e.target.closest(".answer-tag");
    if (!tag) return;
    const promptIdx = parseInt(tag.dataset.prompt);
    const clusterIdx = parseInt(tag.dataset.cluster);
    const memberIdx = parseInt(tag.dataset.member);
    const text = getAnswerText(promptIdx, clusterIdx, memberIdx);
    if (text === null) return;
    appState.dragState = { promptIdx, clusterIdx, text };
    tag.classList.add("dragging");
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", text);
  });

  document.addEventListener("dragend", () => {
    appState.dragState = null;
    clearDropTargets();
    clearDraggedTag();
  });

  document.addEventListener("dragover", (e) => {
    const card = e.target.closest(".cluster-card");
    const drag = appState.dragState;
    if (!card || !drag) return;
    const cardPrompt = parseInt(card.dataset.prompt);
    const cardCluster = parseInt(card.dataset.cluster);
    if (cardPrompt !== drag.promptIdx || cardCluster === drag.clusterIdx) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    clearDropTargets();
    card.classList.add("drop-target");
  });

  document.addEventListener("dragleave", (e) => {
    const card = e.target.closest(".cluster-card");
    if (!card) return;
    const related = e.relatedTarget;
    if (related && card.contains(related)) return;
    card.classList.remove("drop-target");
  });

  document.addEventListener("drop", (e) => {
    const card = e.target.closest(".cluster-card");
    const drag = appState.dragState;
    if (!card || !drag) return;
    const destPrompt = parseInt(card.dataset.prompt);
    const destCluster = parseInt(card.dataset.cluster);
    if (destPrompt !== drag.promptIdx || destCluster === drag.clusterIdx) return;
    e.preventDefault();
    moveSingleAnswer(drag.promptIdx, drag.clusterIdx, destCluster, drag.text);
    appState.dragState = null;
    clearDropTargets();
    clearDraggedTag();
  });
});
