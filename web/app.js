"use strict";

const $ = (id) => document.getElementById(id);
const results = $("results");

// The bearer token, when the server requires one. Kept in memory only, never
// in localStorage, so a shared machine does not leak it between sessions.
let AUTH_TOKEN = "";

function authHeaders(extra) {
  const h = Object.assign({}, extra || {});
  if (AUTH_TOKEN) h["Authorization"] = "Bearer " + AUTH_TOKEN;
  return h;
}

// ---------------------------------------------------------------------------
// fetch helpers
//
// Every response is read as text and parsed defensively. Calling r.json()
// directly is what produced "JSON.parse: unexpected character at line 1 column
// 1": when the body was empty or the server was too busy to send JSON, the
// parse threw and the user saw a parser error instead of the real problem.
// ---------------------------------------------------------------------------

async function readJson(r) {
  let text = "";
  try {
    text = await r.text();
  } catch (_) {
    throw new Error("could not read the response body");
  }

  const trimmed = (text || "").trim();
  if (!trimmed) {
    // No body at all. Say something useful based on the status.
    if (r.status === 503) {
      throw new Error("server busy, too many connections. Wait a moment and retry.");
    }
    if (r.status === 401) throw new Error("authentication required");
    if (r.status === 429) throw new Error("rate limited, wait a moment and retry");
    if (r.status >= 500) {
      throw new Error("server error (" + r.status + "), no details returned");
    }
    throw new Error("empty response from server (" + r.status + ")");
  }

  try {
    return JSON.parse(trimmed);
  } catch (_) {
    // Not JSON. Most likely an HTML error page from a proxy or a crash dump.
    const looksHtml = /^\s*</.test(trimmed);
    const snippet = trimmed.slice(0, 120).replace(/\s+/g, " ");
    throw new Error(
      "server returned " + (looksHtml ? "an HTML page" : "non-JSON data") +
      " (" + r.status + "): " + snippet
    );
  }
}

async function apiFetch(url, options) {
  const opts = options || {};
  opts.headers = authHeaders(opts.headers);

  // A hung request must not hang the UI forever. The AI path can legitimately
  // take 45s+, so the ceiling is generous.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 120000);
  opts.signal = controller.signal;

  let r;
  try {
    r = await fetch(url, opts);
  } catch (e) {
    clearTimeout(timer);
    if (e && e.name === "AbortError") {
      throw new Error("request timed out after 120s");
    }
    // fetch() itself failed: server down, DNS, connection reset.
    throw new Error("cannot reach the server: " + (e.message || e));
  }
  clearTimeout(timer);

  const data = await readJson(r);

  if (r.status === 401) {
    showTokenBox(true);
    throw new Error("authentication required");
  }
  if (r.status === 429) {
    const wait = data.retry_after_seconds;
    throw new Error("rate limited" + (wait ? ", retry in " + wait + "s" : ""));
  }
  if (r.status === 503) {
    throw new Error(data.error || "server busy, wait a moment and retry");
  }
  if (!r.ok) {
    throw new Error(data.error || ("request failed (" + r.status + ")"));
  }
  return data;
}

// ---------------------------------------------------------------------------
// health / status
// ---------------------------------------------------------------------------

async function checkHealth() {
  try {
    const h = await apiFetch("/api/health");
    showTokenBox(false);

    $("tagCount").textContent = (h.tags_loaded || 0).toLocaleString() + " tags";
    if ($("modePill")) {
      $("modePill").textContent = h.mode || "?";
      $("modePill").title =
        "mode " + h.mode +
        " \u00b7 offline capable: " + h.offline_capable +
        " \u00b7 derived structures: " + h.derived_built;
    }

    const dot = $("statusDot");
    const hint = $("aiHint");

    if (!h.ai_enabled) {
      dot.className = "dot warn";
      dot.title = "AI disabled";
      hint.textContent =
        "Running offline. No AI, no network calls, the engine does all the work.";
      hint.className = "hint";
    } else if (h.ai && h.ai.ok) {
      dot.className = "dot ok";
      dot.title = "AI ready: " + (h.ai.detail || "");
      const models = ((h.ai.models || [])).slice(0, 3).join(", ");
      hint.textContent =
        "AI ready (" + h.ai_provider + ")" + (models ? " \u2014 " + models : "");
      hint.className = "hint ok";
    } else {
      dot.className = "dot off";
      dot.title = (h.ai && h.ai.detail) || "unavailable";
      hint.textContent =
        "AI unreachable. The offline engine still works \u2014 untick AI assist, or keep it " +
        "on and results come back engine-only.";
      hint.className = "hint bad";
    }

    const sec = h.security || {};
    $("foot").textContent =
      h.tags_loaded + " tags \u00b7 " + (h.vocabulary_terms || 0) + " terms \u00b7 " +
      "mode " + h.mode + " \u00b7 AI " + (h.ai_enabled ? h.ai_provider : "off") +
      " \u00b7 rate " + sec.rate_limit_per_min + "/min" +
      (sec.auth_required ? " \u00b7 auth on" : "") +
      (sec.tls ? " \u00b7 tls" : "");
  } catch (e) {
    const msg = String(e.message || e);
    if (msg.indexOf("authentication") === -1) {
      $("statusDot").className = "dot off";
      $("foot").textContent = msg;
    }
  }
}

function showTokenBox(show) {
  let box = $("tokenBox");
  if (show && !box) {
    box = document.createElement("div");
    box.id = "tokenBox";
    box.className = "tokenbox";
    box.innerHTML =
      '<label class="lbl sm" for="tok">Access token required</label>' +
      '<div class="tokrow">' +
      '<input id="tok" type="password" placeholder="paste AUTH_TOKEN" autocomplete="off">' +
      '<button id="tokSave" class="primary">Save</button>' +
      "</div>" +
      '<div class="hint">Stored in memory for this tab only. Never written to disk.</div>';
    $("aiHint").parentNode.insertBefore(box, $("aiHint"));
    $("tokSave").addEventListener("click", () => {
      AUTH_TOKEN = $("tok").value.trim();
      $("tok").value = "";
      checkHealth();
    });
    $("tok").addEventListener("keydown", (e) => {
      if (e.key === "Enter") $("tokSave").click();
    });
  } else if (!show && box) {
    box.remove();
  }
}

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------

async function runSearch() {
  const q = $("q").value.trim();
  if (!q) { $("q").focus(); return; }

  const btn = $("go");
  btn.disabled = true;
  const useAI = $("useAI").checked;
  results.innerHTML =
    '<div class="empty"><span class="spin"></span>' +
    (useAI
      ? "Running the offline engine, then asking the AI to re-rank..."
      : "Running the offline engine...") +
    "</div>";

  const body = {
    query: q,
    use_ai: useAI,
    provider: $("provider").value,
    model: $("model").value.trim(),
    include: $("include").value,
    exclude: $("exclude").value,
    categories: $("category").value,
    limit: parseInt($("limit").value, 10) || 40,
  };

  try {
    const data = await apiFetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    render(data);
  } catch (e) {
    renderError(String(e.message || e));
  } finally {
    btn.disabled = false;
  }
}

function renderError(msg) {
  const busy = /busy|rate limited|cannot reach/i.test(msg);
  results.innerHTML =
    '<div class="empty">' +
    "<b>" + (busy ? "Could not run that search" : "Search failed") + "</b><br><br>" +
    esc(msg) +
    (busy
      ? "<br><br>Nothing is broken \u2014 the engine is fine. Wait a second and hit " +
        "<b>Find tags</b> again."
      : "") +
    "</div>";
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const SOURCE_LABEL = {
  engine: "engine",
  ai: "AI only",
  "engine+ai": "engine + AI",
};

function render(d) {
  if (!d.tags || !d.tags.length) {
    results.innerHTML =
      '<div class="empty">No tags matched "<b>' + esc(d.query) + '</b>".<br><br>' +
      "Try plainer words (sad, fast, chill), drop the category filter, " +
      "or untick AI assist to see the raw engine results.</div>";
    return;
  }

  // --- meta bar
  let html = '<div class="meta">';
  html += '<span class="badge eng">' + d.deterministic_count + " engine hits</span>";
  html += '<span class="badge">' + d.elapsed_ms + " ms</span>";
  html += '<span class="badge">mode: ' + esc(d.mode) + "</span>";
  if (d.offline) html += '<span class="badge">offline</span>';
  if (d.ai && d.ai.used) {
    const cached = d.ai.cached ? " (cached)" : "";
    html += '<span class="badge ai">AI: ' + esc(d.ai.model || d.ai.provider) +
            cached + "</span>";
    if (d.ai.latency_ms) {
      html += '<span class="badge">' + d.ai.latency_ms + " ms ai</span>";
    }
  } else if (d.ai && d.ai.error) {
    html += '<span class="badge bad">AI off: ' + esc(d.ai.error) + "</span>";
  }
  html += "</div>";

  // --- offline understanding
  const u = d.understanding || {};
  if (u.summary) {
    const bits = [];
    if (u.energy && u.energy !== "neutral") bits.push("energy: " + u.energy);
    if (u.wants_vocals === "no") bits.push("instrumental only");
    if (u.wants_vocals === "yes") bits.push("wants vocals");
    if (u.use_case) bits.push("use case: " + u.use_case);
    if (u.guessed_categories && Object.keys(u.guessed_categories).length) {
      for (const [w, c] of Object.entries(u.guessed_categories)) {
        bits.push(esc(w) + " \u2192 " + esc(c));
      }
    }
    html += '<div class="understanding"><span class="ulabel">offline reading</span>' +
            esc(u.summary) +
            (bits.length ? '<span class="ubits">' + bits.join(" \u00b7 ") + "</span>" : "") +
            "</div>";
  }

  if (d.ai && d.ai.used && d.ai.reason) {
    html += '<div class="reason">' + esc(d.ai.reason) + "</div>";
  }

  // --- tags
  html += '<div class="taglist">';
  for (const t of d.tags) {
    let cls = "tag";
    if (t.source === "ai") cls = "tag ai";
    else if (t.source === "engine+ai") cls = "tag both";
    else if (t.source && t.source.indexOf("use-case") !== -1) cls = "tag uc";

    const cats = (t.categories || [])
      .map((c) => '<span class="cat">' + esc(c) + "</span>").join("");
    const src = t.source === "ai" || t.source === "engine+ai"
      ? '<span class="src">AI</span>' : "";
    const conf = typeof t.confidence === "number"
      ? '<span class="conf">' + Math.round(t.confidence * 100) + "%</span>" : "";
    const signals = (t.signals || []).length
      ? '<span class="sig">' + t.signals.map(esc).join(",") + "</span>" : "";

    html +=
      '<div class="' + cls + '" data-tag="' + esc(t.tag) +
      '" title="click to copy and add to Must include; shift-click to Never show">' +
      "<span>" + esc(t.tag) + "</span>" + cats + src + signals + conf +
      '<span class="sc">' + t.score.toFixed(1) + "</span></div>";
  }
  html += "</div>";

  // --- palette
  if (d.palette && Object.keys(d.palette).length) {
    html += '<div class="palette"><h3>Related \u2014 click to add</h3>';
    for (const [cat, tags] of Object.entries(d.palette)) {
      if (!tags.length) continue;
      html += '<div class="prow"><span class="pk">' + esc(cat) + "</span>";
      for (const tg of tags) {
        html += '<span class="ptag" data-tag="' + esc(tg) + '">' + esc(tg) + "</span>";
      }
      html += "</div>";
    }
    html += "</div>";
  }

  // --- learned graph neighbours
  if (d.related && Object.keys(d.related).length) {
    html += '<div class="palette"><h3>Learned neighbours \u2014 from shared tokens, no model</h3>';
    for (const [seed, rel] of Object.entries(d.related)) {
      html += '<div class="prow"><span class="pk">' + esc(seed) + "</span>";
      for (const r of rel.slice(0, 5)) {
        const shared = (r.shared || []).join("+");
        html += '<span class="ptag" data-tag="' + esc(r.tag) + '" title="shares: ' +
                esc(shared) + '">' + esc(r.tag) + "</span>";
      }
      html += "</div>";
    }
    html += "</div>";
  }

  // --- hallucination guard report
  if (d.ai && d.ai.dropped_hallucinations && d.ai.dropped_hallucinations.length) {
    html +=
      '<div class="dropped"><b>Blocked ' + d.ai.dropped_hallucinations.length +
      " tag(s) the AI made up</b> (not in your vocabulary, so they were discarded): " +
      d.ai.dropped_hallucinations.map((x) => "<code>" + esc(x) + "</code>").join(", ") +
      "</div>";
  }

  results.innerHTML = html;

  results.querySelectorAll("[data-tag]").forEach((el) => {
    el.addEventListener("click", async (ev) => {
      const tag = el.dataset.tag;
      const input = ev.shiftKey ? $("exclude") : $("include");
      const cur = input.value.trim();
      const parts = cur ? cur.split(",").map((s) => s.trim()).filter(Boolean) : [];
      if (!parts.some((p) => p.toLowerCase() === tag.toLowerCase())) parts.push(tag);
      input.value = parts.join(", ");
      try {
        await navigator.clipboard.writeText(tag);
        el.classList.add("copied");
        setTimeout(() => el.classList.remove("copied"), 1100);
      } catch (_) { /* clipboard blocked; the include box still updated */ }
    });
  });
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------

$("go").addEventListener("click", runSearch);
$("q").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runSearch();
});
document.querySelectorAll(".ex").forEach((b) =>
  b.addEventListener("click", () => { $("q").value = b.dataset.q; runSearch(); })
);
$("provider").addEventListener("change", () => {
  const k = $("provider").value === "keyless";
  $("model").placeholder = k ? "blank = openai-fast" : "blank = llama3.2 (local)";
  checkHealth();
});

checkHealth();
setInterval(checkHealth, 60000);
