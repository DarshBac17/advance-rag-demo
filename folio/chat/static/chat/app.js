/* Folio front end: vanilla JS against the Django JSON API. */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const boot = JSON.parse($("#boot").textContent);

  const state = {
    docs: boot.documents || [],
    selected: new Set(),
    mode: "advanced",
    turns: [],
    current: null,          // {turn, key} of the answer shown in the evidence pane
    pollTimer: null,
    activeDoc: null,
  };

  // ---------------------------------------------------------------- helpers
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const cookie = (n) => (document.cookie.match(new RegExp("(^|; )" + n + "=([^;]*)")) || [])[2];
  async function api(url, opts = {}) {
    const o = { ...opts, headers: { "X-CSRFToken": cookie("csrftoken") || "", ...(opts.headers || {}) } };
    if (o.json !== undefined) { o.body = JSON.stringify(o.json); o.headers["Content-Type"] = "application/json"; delete o.json; }
    const r = await fetch(url, o);
    let data = {};
    try { data = await r.json(); } catch (_) { /* image or empty */ }
    if (!r.ok) throw new Error(data.error || `Request failed (${r.status})`);
    return data;
  }
  let toastT;
  function toast(msg) {
    const t = $("#toast"); t.textContent = msg; t.classList.add("is-on");
    clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("is-on"), 3600);
  }
  const fmtPages = (p) => !p || !p.length ? "" : p.length === 1 ? `p. ${p[0]}` : `pp. ${p[0]}–${p[p.length - 1]}`;
  const kindLabel = { text: "Text", table: "Table", figure: "Chart labels", vision: "Chart (vision)", caption: "Caption", section: "Section", naive: "Fixed-size chunk" };
  const ready = () => state.docs.filter((d) => d.status === "ready");

  // ---------------------------------------------------------------- health
  async function loadHealth() {
    try {
      const h = await api("/api/health");
      const el = $("#health");
      if (h.warnings.length) { el.innerHTML = h.warnings.map((w) => `<p>${esc(w)}</p>`).join(""); el.hidden = false; }
    } catch (e) { /* non-fatal */ }
  }

  // ---------------------------------------------------------------- documents
  function renderDocs() {
    const ul = $("#doclist");
    if (!state.docs.length) {
      ul.innerHTML = `<li class="muted" style="font-size:13px">No documents yet. Add a PDF to start.</li>`;
    } else {
      ul.innerHTML = state.docs.map((d) => {
        const busy = d.status === "queued" || d.status === "processing";
        const s = d.stats || {};
        const meta = d.status === "ready"
          ? `<span class="doc-strategy">${esc(d.strategy_label || d.profile)}</span><span>${d.pages} pages</span><span>${s.chunks} chunks</span>${s.tables ? `<span>${s.tables} tables</span>` : ""}`
          : busy ? `<span>${esc(d.message || d.stage)}</span>` : "";
        return `<li class="doc ${state.activeDoc === d.id ? "is-active" : ""}" data-id="${d.id}">
          <input type="checkbox" aria-label="Include ${esc(d.title)}" ${state.selected.has(d.id) ? "checked" : ""} ${d.status !== "ready" ? "disabled" : ""}>
          <button type="button" class="doc-title">${esc(d.title)}</button>
          <div class="doc-meta">${meta}</div>
          ${busy ? `<div class="doc-progress" role="progressbar" aria-valuenow="${d.progress}" aria-valuemin="0" aria-valuemax="100"><i style="width:${d.progress}%"></i></div>` : ""}
          ${d.status === "error" ? `<div class="doc-error">${esc(d.message)}</div>` : ""}
        </li>`;
      }).join("");
    }
    renderScope();
    const busy = state.docs.some((d) => d.status === "queued" || d.status === "processing");
    if (busy && !state.pollTimer) state.pollTimer = setInterval(pollDocs, 1200);
    if (!busy && state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
  }
  async function pollDocs() {
    const before = new Set(ready().map((d) => d.id));
    const { documents } = await api("/api/documents");
    state.docs = documents;
    const newly = ready().filter((d) => !before.has(d.id));
    newly.forEach((d) => toast(`${d.title} is ready: read as ${d.strategy_label}.`));
    renderDocs();
    if (newly.length) loadSamples();
  }
  function renderScope() {
    const ids = scopeIds();
    const names = state.docs.filter((d) => ids.includes(d.id)).map((d) => d.title);
    $("#scope").innerHTML = !ids.length ? "No ready documents yet"
      : state.selected.size ? `Asking <b>${esc(names.join(", "))}</b>`
      : `Asking <b>all ${ids.length} ready document${ids.length > 1 ? "s" : ""}</b>`;
  }
  const scopeIds = () => (state.selected.size ? [...state.selected] : ready().map((d) => d.id));

  $("#doclist").addEventListener("change", (e) => {
    const li = e.target.closest(".doc"); if (!li) return;
    e.target.checked ? state.selected.add(li.dataset.id) : state.selected.delete(li.dataset.id);
    renderScope(); loadSamples();
  });
  $("#doclist").addEventListener("click", (e) => {
    const b = e.target.closest(".doc-title"); if (!b) return;
    showDoc(b.closest(".doc").dataset.id);
  });

  async function upload(file) {
    if (!file) return;
    if (!/\.pdf$/i.test(file.name)) { toast("Only PDF files can be indexed."); return; }
    const fd = new FormData(); fd.append("file", file); fd.append("profile", $("#profile").value);
    toast(`Uploading ${file.name}…`);
    try {
      const { document } = await api("/api/documents", { method: "POST", body: fd });
      state.docs.push(document); renderDocs();
    } catch (e) { toast(e.message); }
  }
  $("#file").addEventListener("change", (e) => { upload(e.target.files[0]); e.target.value = ""; });
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
  drop.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));

  // ---------------------------------------------------------------- document inspector
  async function showDoc(id) {
    state.activeDoc = id; renderDocs(); openTab("doc"); showPane("pane-evidence");
    const el = $("#tab-doc"); el.innerHTML = `<p class="pending">Loading</p>`;
    try {
      const { document: d } = await api(`/api/documents/${id}`);
      const s = d.stats || {};
      const outline = (d.outline || []).filter((o) => o.depth <= 1).slice(0, 40);
      const aliases = Object.entries(d.aliases || {}).slice(0, 16);
      el.innerHTML = `
        <h3 style="font:600 18px/1.3 var(--serif)">${esc(d.title)}</h3>
        <p style="font-size:13px;color:var(--graphite);margin:4px 0 0">${esc(d.filename)}</p>
        ${d.status === "ready" ? `
        <p class="subhead">Read as: ${esc(d.strategy_label)}</p>
        <p style="font-size:13px;margin:0">${esc(d.why || "")}</p>
        <dl class="kv">
          <dt>Detected because</dt><dd>${esc((d.reasons || []).join("; "))}</dd>
          <dt>Pages</dt><dd>${d.pages}</dd>
          <dt>Chunks</dt><dd>${s.chunks} (${Object.entries(s.kinds || {}).map(([k, v]) => `${v} ${kindLabel[k] ? kindLabel[k].toLowerCase() : k}`).join(", ")})</dd>
          ${s.parents ? `<dt>Parent sections</dt><dd>${s.parents}</dd>` : ""}
          <dt>Footnotes linked</dt><dd>${s.notes}</dd>
          <dt>Chart pages described</dt><dd>${s.vision}${d.vision_note ? ` — ${esc(d.vision_note)}` : ""}</dd>
          <dt>Knowledge graph</dt><dd>${s.graph_nodes} nodes, ${s.graph_edges} edges, ${s.entities} entities</dd>
          ${s.ligature_fixes ? `<dt>Text repairs</dt><dd>${s.ligature_fixes} damaged word forms fixed (e.g. “Te” → “The”)</dd>` : ""}
          ${s.skipped && Object.keys(s.skipped).length ? `<dt>Not indexed</dt><dd>${Object.entries(s.skipped).map(([k, v]) => `${v} ${k.replace(/_/g, " ")}`).join(", ")}</dd>` : ""}
          <dt>Embeddings</dt><dd>${esc(d.embedder?.name || "")}${d.embedder && !d.embedder.semantic ? " (offline fallback)" : ""}</dd>
          <dt>Indexed in</dt><dd>${s.seconds} s</dd>
        </dl>
        ${aliases.length ? `<p class="subhead">Abbreviations found</p><dl class="kv">${aliases.map(([a, b]) => `<dt>${esc(a.toUpperCase())}</dt><dd>${esc(b)}</dd>`).join("")}</dl>` : ""}
        ${outline.length ? `<p class="subhead">Sections questions can be routed to</p><ul class="outline">${outline.map((o) => `<li style="margin-left:${o.depth * 14}px">${esc(o.name)}</li>`).join("")}</ul>` : ""}`
        : `<p>${esc(d.message || d.status)}</p>`}
        <p class="subhead">Re-index</p>
        <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
          <select id="reindex-profile" style="padding:6px;border:1px solid var(--rule);border-radius:6px;background:var(--surface)">
            <option value="">Detect automatically</option><option value="narrative">Plain text</option>
            <option value="long_structured">Long, structured</option><option value="visual">Visual</option>
          </select>
          <button type="button" class="linkish" id="reindex">Re-index with this strategy</button>
          <button type="button" class="linkish" id="delete-doc" style="color:var(--bad)">Delete document</button>
        </div>`;
      $("#reindex").onclick = async () => {
        await api(`/api/documents/${id}`, { method: "POST", json: { profile: $("#reindex-profile").value } });
        toast("Re-indexing started."); await pollDocs();
      };
      $("#delete-doc").onclick = async () => {
        if (!confirm(`Delete “${d.title}” and its index?`)) return;
        await api(`/api/documents/${id}`, { method: "DELETE" });
        state.docs = state.docs.filter((x) => x.id !== id); state.selected.delete(id);
        el.innerHTML = `<p class="muted">Deleted.</p>`; renderDocs(); loadSamples();
      };
    } catch (e) { el.innerHTML = `<p>${esc(e.message)}</p>`; }
  }

  // ---------------------------------------------------------------- samples
  async function loadSamples() {
    const box = $("#samples");
    try {
      const { samples } = await api("/api/samples");
      const ids = scopeIds();
      const qs = [...new Set(ids.flatMap((id) => samples[id] || []))].slice(0, 6);
      box.innerHTML = qs.map((q) => `<button type="button" class="sample">${esc(q)}</button>`).join("");
    } catch (_) { box.innerHTML = ""; }
  }
  $("#samples").addEventListener("click", (e) => {
    const b = e.target.closest(".sample"); if (!b) return;
    $("#q").value = b.textContent; $("#q").focus();
  });

  // ---------------------------------------------------------------- asking
  $$(".modes button").forEach((b) => b.addEventListener("click", () => {
    state.mode = b.dataset.mode;
    $$(".modes button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
  }));
  $("#q").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
  });
  $("#composer").addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("#q").value.trim(); if (!q) return;
    if (!ready().length) { toast("Upload a document and wait until it is ready."); return; }
    const turn = { q, mode: state.mode, docIds: scopeIds(), result: null, pending: true };
    state.turns.push(turn); $("#q").value = ""; $("#send").disabled = true;
    renderThread();
    try {
      const r = await api("/api/ask", { method: "POST", json: { question: q, doc_ids: turn.docIds, mode: turn.mode } });
      turn.result = r.result;
    } catch (err) { turn.error = err.message; }
    turn.pending = false; $("#send").disabled = false;
    renderThread();
    if (turn.result) selectAnswer(state.turns.length - 1, turn.mode === "compare" ? "advanced" : null);
  });

  const numbersIn = (txt) => [...new Set((txt.replace(/\[\d+\]/g, " ").match(/\d[\d,]*(?:\.\d+)?/g) || [])
    .filter((n) => n.replace(/,/g, "").length >= 2 || /\./.test(n)))];

  function answerHTML(res, label, turnIdx, key) {
    const paras = esc(res.answer).split(/\n{2,}|\n/).filter(Boolean)
      .map((p) => `<p>${p.replace(/\[(\d+)\]/g, (m, n) => `<button type="button" class="cite" data-n="${n}" data-turn="${turnIdx}" data-key="${key || ""}" aria-label="Show passage ${n}">${n}</button>`)}</p>`).join("");
    const v = res.verification || {};
    const vtext = { grounded: "Grounded", refused: "Not in the document", uncited: "No citations", "check numbers": "Check numbers" }[v.verdict] || v.verdict;
    const vclass = { grounded: "v-grounded", refused: "v-refused", uncited: "v-uncited", "check numbers": "v-check" }[v.verdict] || "";
    const extra = v.unsupported_numbers && v.unsupported_numbers.length ? `<span>${esc(v.unsupported_numbers.join(", "))} not found in the cited passages</span>` : "";
    return `<div class="answer" data-turn="${turnIdx}" data-key="${key || ""}" tabindex="0">
      ${label ? `<span class="answer-label">${esc(label)}</span>` : ""}
      ${paras}
      <div class="verdict"><span class="v ${vclass}">${esc(vtext)}</span>${extra}
        <span>${res.passages.length} passages</span><span>${(res.ms / 1000).toFixed(1)} s</span>
        ${res.method === "extractive" ? "<span>extractive (no LLM key)</span>" : ""}
        ${res.attempts > 1 ? "<span>retried once</span>" : ""}</div>
    </div>`;
  }

  function renderThread() {
    const th = $("#thread");
    $("#empty")?.remove();
    th.innerHTML = state.turns.map((t, i) => {
      const modeName = { advanced: "Folio pipeline", naive: "Basic RAG", compare: "Compare" }[t.mode];
      let body = "";
      if (t.pending) body = `<div class="pending">Reading the documents</div>`;
      else if (t.error) body = `<div class="answer" style="cursor:default;color:var(--bad)">${esc(t.error)}</div>`;
      else if (t.mode === "compare") body = `<div class="answers is-compare">${answerHTML(t.result.naive, "Basic RAG", i, "naive")}${answerHTML(t.result.advanced, "Folio pipeline", i, "advanced")}</div>`;
      else body = `<div class="answers">${answerHTML(t.result, "", i, null)}</div>`;
      return `<article class="turn"><div class="question">${esc(t.q)}<small>${modeName}</small></div>${body}</article>`;
    }).join("") || "";
    th.scrollTop = th.scrollHeight;
    markSelected();
  }

  function resultFor(turnIdx, key) {
    const t = state.turns[turnIdx]; if (!t || !t.result) return null;
    return key ? t.result[key] : t.result;
  }
  function markSelected() {
    $$(".answer").forEach((a) => a.classList.toggle("is-selected",
      state.current && +a.dataset.turn === state.current.turn && (a.dataset.key || null) === (state.current.key || null)));
  }

  $("#thread").addEventListener("click", (e) => {
    const c = e.target.closest(".cite");
    if (c) {
      const key = c.dataset.key || null;
      selectAnswer(+c.dataset.turn, key);
      openTab("passages"); showPane("pane-evidence");
      const p = $(`#passage-${c.dataset.n}`);
      if (p) { p.scrollIntoView({ block: "start", behavior: "smooth" }); p.classList.remove("is-flash"); void p.offsetWidth; p.classList.add("is-flash"); }
      return;
    }
    const a = e.target.closest(".answer[data-turn]");
    if (a) selectAnswer(+a.dataset.turn, a.dataset.key || null);
  });

  // ---------------------------------------------------------------- evidence pane
  function selectAnswer(turnIdx, key) {
    const res = resultFor(turnIdx, key); if (!res) return;
    state.current = { turn: turnIdx, key };
    markSelected();
    renderPassages(res); renderTrace(res); renderGraph(res);
  }

  function highlight(text, nums) {
    let h = esc(text);
    h = h.replace(/(\[Note \d+\][^\n]*)/g, '<span class="fn">$1</span>').replace(/(\[note \d+\])/g, '<span class="fn">$1</span>');
    nums.sort((a, b) => b.length - a.length).forEach((n) => {
      const re = new RegExp(`(?<![\\d.,])(${n.replace(/[.,]/g, (c) => "\\" + c)})(?![\\d])`, "g");
      h = h.replace(re, "<mark>$1</mark>");
    });
    return h;
  }

  function renderPassages(res) {
    const el = $("#tab-passages");
    const cited = new Set((res.verification?.cited) || []);
    const nums = numbersIn(res.answer);
    if (!res.passages.length) { el.innerHTML = `<p class="muted">Nothing was retrieved for this question.</p>`; return; }
    el.innerHTML = res.passages.map((p, i) => {
      const n = i + 1;
      const ranks = Object.entries(p.ranks || {}).map(([k, v]) => `${k.replace(/_?(\d)$/, "").replace("_", " ")} #${v}`).join(", ");
      return `<div class="passage ${cited.has(n) ? "is-cited" : ""}" id="passage-${n}">
        <div class="phead"><span class="pnum">${n}</span><span class="kind kind-${esc(p.kind)}">${esc(kindLabel[p.kind] || p.kind)}</span>
          <span>${esc(fmtPages(p.pages))}</span><span>${esc(p.doc_title)}</span>
          ${p.section_path?.length ? `<span class="ppath">${esc(p.section_path.join(" › "))}</span>` : ""}</div>
        <div class="ptext">${highlight(p.text, nums)}</div>
        <div class="pfoot"><span>${p.via && p.via.startsWith("graph") ? esc(p.via.replace("graph:", "Added by graph:")) : "Found by search"}</span>
          ${ranks ? `<span>${esc(ranks)}</span>` : ""}${p.rerank != null ? `<span>rerank ${p.rerank}</span>` : ""}
          <button type="button" class="linkish showpage" data-doc="${p.doc_id}" data-page="${p.pages[0]}">Show page ${p.pages[0]}</button></div>
      </div>`;
    }).join("");
  }
  $("#tab-passages").addEventListener("click", (e) => {
    const b = e.target.closest(".showpage"); if (!b) return;
    const box = b.closest(".passage");
    const old = $(".pageimg", box);
    if (old) { old.remove(); b.textContent = `Show page ${b.dataset.page}`; return; }
    const img = new Image(); img.className = "pageimg"; img.alt = `Page ${b.dataset.page} of the source PDF`;
    img.loading = "lazy"; img.src = `/api/documents/${b.dataset.doc}/page/${b.dataset.page}.png`;
    box.appendChild(img); b.textContent = "Hide page";
  });

  function renderTrace(res) {
    const el = $("#tab-trace");
    const a = res.analysis || {};
    const analysis = Object.entries(a).map(([doc, x]) => `
      <dl class="kv"><dt>Document</dt><dd>${esc(state.docs.find((d) => d.id === doc)?.title || doc)}</dd>
        <dt>Understood by</dt><dd>${x.method === "llm" ? "LLM (fast model)" : "rules over the document outline"}</dd>
        <dt>Section filter</dt><dd>${x.sections?.length ? esc(x.sections.join(", ")) + " (soft: fused with unfiltered results)" : "none"}</dd>
        <dt>Search query</dt><dd>${esc(x.search_query)}</dd>
        ${x.hypothetical_answer ? `<dt>Hypothetical answer</dt><dd>${esc(x.hypothetical_answer)}</dd>` : ""}
        ${x.sub_questions?.length ? `<dt>Sub-questions</dt><dd>${x.sub_questions.map(esc).join("<br>")}</dd>` : ""}</dl>`).join("");
    el.innerHTML = `<ol class="steps">${res.trace.map((t) => `<li><span class="stage">${esc(t.stage)}</span><span class="ms">${t.ms} ms</span><span class="stxt">${esc(t.text)}</span></li>`).join("")}</ol>
      ${analysis ? `<p class="subhead">Query understanding</p>${analysis}` : ""}`;
  }

  function renderGraph(res) {
    const el = $("#tab-graph");
    const g = res.graph || { nodes: [], edges: [] };
    if (!g.nodes.length) { el.innerHTML = `<p class="muted">${res.pipeline === "naive" ? "Basic RAG does not use the knowledge graph." : "No graph links for this answer."}</p>`; return; }
    const cols = { entity: 0, chunk: 1, section: 2, note: 2 };
    const color = { entity: "var(--k-note)", chunk: "var(--pencil)", section: "var(--k-table)", note: "var(--k-figure)" };
    const byCol = [[], [], []];
    const seen = new Set();
    g.nodes.forEach((n) => { if (!seen.has(n.id)) { seen.add(n.id); byCol[cols[n.type] ?? 1].push(n); } });
    const W = 380, rowH = 28, h = Math.max(...byCol.map((c) => c.length)) * rowH + 30;
    const xs = [10, 140, 250], pos = {};
    byCol.forEach((c, ci) => c.forEach((n, ri) => { pos[n.id] = { x: xs[ci], y: 22 + ri * rowH }; }));
    const lines = g.edges.filter(([a, b]) => pos[a] && pos[b]).map(([a, b]) => {
      const p = pos[a], q = pos[b];
      return `<path class="e" d="M${p.x + 6},${p.y} C${(p.x + q.x) / 2},${p.y} ${(p.x + q.x) / 2},${q.y} ${q.x},${q.y}"/>`;
    }).join("");
    const nodes = Object.entries(pos).map(([id, p]) => {
      const n = g.nodes.find((x) => x.id === id);
      const label = (n.label || id).slice(0, 26);
      return `<g><circle cx="${p.x + 3}" cy="${p.y}" r="4.5" fill="${color[n.type]}"/><text x="${p.x + 12}" y="${p.y + 4}">${esc(label)}</text><title>${esc(n.label)}${n.how ? " — " + esc(n.how) : ""}</title></g>`;
    }).join("");
    el.innerHTML = `<div class="legend"><span><i style="background:var(--k-note)"></i>Entities in the question</span><span><i style="background:var(--pencil)"></i>Passages used</span><span><i style="background:var(--k-table)"></i>Sections</span><span><i style="background:var(--k-figure)"></i>Footnotes</span></div>
      <svg class="graph-svg" viewBox="0 0 ${W} ${h}" role="img" aria-label="Knowledge graph links behind this answer">${lines}${nodes}</svg>
      <p style="font-size:13px;color:var(--graphite)">Entities named in the question (and their abbreviations) point to the passages that mention them; passages point to their sections and the footnotes they cite.</p>`;
  }

  // ---------------------------------------------------------------- tabs, panes, views
  function openTab(name) {
    $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    $$(".tabpanel").forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
  }
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => openTab(b.dataset.tab)));
  function showPane(id) {
    $$(".pane").forEach((p) => p.classList.toggle("is-shown", p.id === id));
    $$(".mobile-nav button").forEach((b) => b.classList.toggle("is-on", b.dataset.pane === id));
  }
  $$(".mobile-nav button").forEach((b) => b.addEventListener("click", () => showPane(b.dataset.pane)));
  $$(".view-btn").forEach((b) => b.addEventListener("click", () => {
    $$(".view-btn").forEach((x) => x.classList.toggle("is-on", x === b));
    $("#view-chat").hidden = b.dataset.view !== "chat"; $("#view-eval").hidden = b.dataset.view !== "eval";
    showPane("pane-convo");
  }));

  // ---------------------------------------------------------------- evaluation
  const pct = (x) => x == null ? "–" : (x * 100).toFixed(0) + "%";
  const num = (x) => x == null ? "–" : x.toFixed(2);
  function evalTable(r) {
    const n = r.metrics.naive, a = r.metrics.advanced;
    const row = (label, k, f) => {
      const better = a && n && a[k] != null && n[k] != null && a[k] > n[k] ? "better" : "";
      return `<tr><td>${label}</td><td>${f(n?.[k])}</td><td class="${better}">${f(a?.[k])}</td></tr>`;
    };
    const qrows = r.questions.map((q) => `<tr class="qrow"><td>${esc(q.question)}<br><span class="muted">${esc(q.type)} · gold p. ${esc((q.gold_pages || []).join(", ") || "none")}</span></td>
      <td>${q.naive ? `rank ${q.naive.rank ?? "–"} <span class="${q.naive.correct ? "yes" : "no"}">${q.naive.correct ? "✓" : "✗"}</span>` : "–"}</td>
      <td>${q.advanced ? `rank ${q.advanced.rank ?? "–"} <span class="${q.advanced.correct ? "yes" : "no"}">${q.advanced.correct ? "✓" : "✗"}</span>` : "–"}</td></tr>`).join("");
    return `<h3 style="font:600 17px/1.3 var(--serif);margin:0 0 8px">${esc(r.title)} <span class="muted" style="font:13px var(--sans)">${r.n} questions · read as ${esc(r.profile)}</span></h3>
      <div class="tablewrap"><table class="score"><thead><tr><th>Metric</th><th>Basic RAG</th><th>Folio pipeline</th></tr></thead><tbody>
      ${row("Answer passage ranked first (hit@1)", "hit@1", pct)}${row("In the top 3 (hit@3)", "hit@3", pct)}${row("In the top 5 (hit@5)", "hit@5", pct)}
      ${row("Mean reciprocal rank", "MRR", num)}${row("Answers correct", "answers", pct)}</tbody></table></div>
      <details class="qs"><summary>Per-question results</summary><div class="tablewrap"><table class="score"><thead><tr><th>Question</th><th>Basic RAG</th><th>Folio</th></tr></thead><tbody>${qrows}</tbody></table></div></details>`;
  }
  $("#run-eval").addEventListener("click", async () => {
    const out = $("#eval-out"), btn = $("#run-eval");
    const docs = ready(); if (!docs.length) { toast("No ready documents to evaluate."); return; }
    btn.disabled = true; out.innerHTML = "";
    for (const d of docs) {
      const slot = document.createElement("div"); slot.innerHTML = `<p class="pending">Evaluating ${esc(d.title)}</p>`; out.appendChild(slot);
      try {
        const res = await api("/api/evaluate", { method: "POST", json: { doc_id: d.id } });
        slot.innerHTML = res.documents.length ? res.documents.map(evalTable).join("")
          : `<p class="muted">${esc(d.title)}: no checked questions for this document.</p>`;
        if (res.note) slot.insertAdjacentHTML("beforeend", `<p class="muted" style="font-size:13px">${esc(res.note)}</p>`);
      } catch (e) { slot.innerHTML = `<p style="color:var(--bad)">${esc(d.title)}: ${esc(e.message)}</p>`; }
    }
    btn.disabled = false;
  });

  // ---------------------------------------------------------------- start
  async function loadHistory() {
    try {
      const { messages } = await api("/api/history");
      state.turns = messages.map((m) => ({ q: m.question, mode: m.mode, docIds: m.doc_ids, result: m.result }));
      if (state.turns.length) { renderThread(); }
    } catch (_) { /* fresh start */ }
  }
  showPane("pane-convo");
  renderDocs(); loadHealth(); loadSamples(); loadHistory();
})();
