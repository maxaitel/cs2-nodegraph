import cytoscape from "cytoscape";
import Papa from "papaparse";
import { createIcons, ImageDown, RotateCcw, Scan, Search } from "lucide";
import "./styles.css";

const colors = {
  ace: "#d7263d",
  flick: "#6f4bd8",
  clutch: "#f28e2b",
  "1v1": "#2a9d8f",
  impressive_multikill: "#4e79a7",
  other: "#8a8f98",
};

const labels = ["ace", "flick", "clutch", "1v1", "impressive_multikill", "other"];
const noFlickLabels = labels.filter((label) => label !== "flick");
const labelText = {
  ace: "ace",
  flick: "flick",
  clutch: "clutch",
  "1v1": "1v1",
  impressive_multikill: "impressive",
  other: "other",
};

const state = {
  nodes: [],
  edges: [],
  summary: null,
  activeLabels: new Set(labels),
  query: "",
  edgeThreshold: 0.35,
  layout: "preset",
  flickMode: "show",
  selectedId: null,
  selectedNode: null,
  cy: null,
};

const byId = (id) => document.getElementById(id);
const els = {
  meta: byId("meta"),
  legend: byId("legend"),
  search: byId("search"),
  layout: byId("layout"),
  flickMode: byId("flick-mode"),
  edgeStrength: byId("edge-strength"),
  playViewer: byId("play-viewer"),
  selection: byId("selection"),
  groupCounts: byId("group-counts"),
  runInfo: byId("run-info"),
  fit: byId("fit"),
  reset: byId("reset"),
  exportPng: byId("export-png"),
};

createIcons({ icons: { ImageDown, RotateCcw, Scan, Search } });

function numberValue(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  })[ch]);
}

async function loadCsv(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}: ${response.status}`);
  }
  const text = await response.text();
  const parsed = Papa.parse(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });
  if (parsed.errors.length) {
    throw new Error(parsed.errors.map((err) => err.message).join("; "));
  }
  return parsed.data;
}

async function loadData() {
  const [nodes, edges, summaryResponse] = await Promise.all([
    loadCsv("/data/opencs2_play_vectors.csv"),
    loadCsv("/data/opencs2_play_edges.csv"),
    fetch("/data/opencs2_play_summary.json"),
  ]);
  if (!summaryResponse.ok) {
    throw new Error(`Could not load summary: ${summaryResponse.status}`);
  }
  state.nodes = nodes.map((node, index) => normalizeNode(node, index));
  state.edges = edges.map((edge, index) => normalizeEdge(edge, index));
  state.summary = await summaryResponse.json();
}

function normalizeNode(node, index) {
  const tags = String(node.tags || node.primary_label || "other")
    .split(";")
    .map((tag) => tag.trim())
    .filter(Boolean);
  const basePrimaryLabel = labels.includes(node.primary_label) ? node.primary_label : "other";
  const noFlickPrimaryLabel = noFlickPrimaryFor(node, tags);
  return {
    ...node,
    index,
    cyId: `n${index}`,
    play_id: String(node.play_id),
    primary_label: basePrimaryLabel,
    base_primary_label: basePrimaryLabel,
    no_flick_primary_label: noFlickPrimaryLabel,
    tags,
    x: numberValue(node.x),
    y: numberValue(node.y),
    interest_score: numberValue(node.interest_score),
    similarity_text: tags.map((tag) => labelText[tag] || tag).join(", "),
  };
}

function noFlickPrimaryFor(node, tags) {
  if (tags.includes("ace")) {
    return "ace";
  }
  if (tags.includes("impressive_multikill") && numberValue(node.player_round_kills) >= 4) {
    return "impressive_multikill";
  }
  if (tags.includes("1v1")) {
    return "1v1";
  }
  if (tags.includes("clutch")) {
    return "clutch";
  }
  if (tags.includes("impressive_multikill")) {
    return "impressive_multikill";
  }
  return "other";
}

function labelsForMode() {
  return state.flickMode === "show" ? labels : noFlickLabels;
}

function effectiveLabelFor(node) {
  return state.flickMode === "relabel" ? node.no_flick_primary_label : node.base_primary_label;
}

function displayTagsFor(node) {
  return state.flickMode === "show" ? node.tags : node.tags.filter((tag) => tag !== "flick");
}

function normalizeEdge(edge, index) {
  return {
    ...edge,
    index,
    cyId: `e${index}`,
    source: Number(edge.source),
    target: Number(edge.target),
    similarity: numberValue(edge.similarity),
    weight: numberValue(edge.weight, 0.2),
  };
}

function graphPositions(nodes) {
  const xs = nodes.map((node) => node.x);
  const ys = nodes.map((node) => node.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = Math.max(0.001, maxX - minX);
  const spanY = Math.max(0.001, maxY - minY);
  return new Map(nodes.map((node) => [
    node.cyId,
    {
      x: ((node.x - minX) / spanX) * 1600,
      y: ((node.y - minY) / spanY) * 1000,
    },
  ]));
}

function graphElements() {
  const positions = graphPositions(state.nodes);
  const nodeElements = state.nodes.map((node) => ({
    group: "nodes",
    data: {
      id: node.cyId,
      playId: node.play_id,
      label: effectiveLabelFor(node),
      baseLabel: node.base_primary_label,
      noFlickLabel: node.no_flick_primary_label,
      tags: node.tags.join(" "),
      map: node.map_name,
      weapon: node.weapon,
      weaponClass: node.weapon_class,
      round: node.round,
      player: node.player_slot,
      score: node.interest_score,
      radius: Math.max(8, Math.min(24, 9 + node.interest_score * 1.15)),
      search: [
        node.play_id,
        node.primary_label,
        node.tags.join(" "),
        node.map_name,
        node.weapon,
        node.weapon_class,
        node.round,
        node.player_slot,
      ].join(" ").toLowerCase(),
    },
    classes: [`label-${node.primary_label.replaceAll("_", "-")}`],
    position: positions.get(node.cyId),
  }));

  const edgeElements = state.edges.map((edge) => ({
    group: "edges",
    data: {
      id: edge.cyId,
      source: `n${edge.source}`,
      target: `n${edge.target}`,
      similarity: edge.similarity,
      weight: edge.weight,
    },
  }));

  return [...nodeElements, ...edgeElements];
}

function makeCy() {
  state.cy = cytoscape({
    container: byId("cy"),
    elements: graphElements(),
    minZoom: 0.08,
    maxZoom: 5,
    wheelSensitivity: 0.18,
    layout: { name: "preset", fit: true, padding: 42 },
    style: [
      {
        selector: "node",
        style: {
          width: "data(radius)",
          height: "data(radius)",
          "background-color": (ele) => colors[ele.data("label")] || colors.other,
          "border-color": "#ffffff",
          "border-width": 1.4,
          "overlay-padding": 4,
          "transition-property": "opacity, border-width, width, height",
          "transition-duration": "120ms",
        },
      },
      {
        selector: "edge",
        style: {
          width: (ele) => Math.max(0.45, ele.data("weight") * 2.1),
          "line-color": "#98a2b3",
          opacity: (ele) => Math.max(0.05, Math.min(0.36, ele.data("weight") * 0.34)),
          "curve-style": "haystack",
          "haystack-radius": 0.2,
        },
      },
      {
        selector: "node:selected",
        style: {
          "border-color": "#111827",
          "border-width": 3,
          width: (ele) => ele.data("radius") + 5,
          height: (ele) => ele.data("radius") + 5,
        },
      },
      {
        selector: ".dimmed",
        style: {
          opacity: 0.08,
        },
      },
      {
        selector: ".neighbor",
        style: {
          opacity: 0.94,
        },
      },
      {
        selector: ".hidden",
        style: {
          display: "none",
        },
      },
    ],
  });

  state.cy.on("tap", "node", (event) => {
    state.selectedId = event.target.id();
    showSelection(event.target);
    emphasizeNeighborhood(event.target);
  });

  state.cy.on("tap", (event) => {
    if (event.target === state.cy) {
      clearSelection();
    }
  });

  const resizeGraph = () => {
    state.cy.resize();
    state.cy.fit(state.cy.elements(":visible"), 42);
  };
  requestAnimationFrame(resizeGraph);
  window.addEventListener("resize", () => {
    state.cy.resize();
  });
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => {
      state.cy.resize();
    });
    observer.observe(byId("cy"));
  }
}

function runLayout(name = state.layout) {
  const common = { fit: true, padding: 42, animate: true, animationDuration: 350 };
  if (name === "preset") {
    state.cy.layout({ name: "preset", ...common }).run();
  } else if (name === "cose") {
    state.cy.layout({
      name: "cose",
      ...common,
      randomize: false,
      nodeRepulsion: 7000,
      idealEdgeLength: 52,
      edgeElasticity: 80,
      numIter: 800,
    }).run();
  } else {
    state.cy.layout({ name, ...common }).run();
  }
}

function nodeMatches(node) {
  if (state.flickMode === "hide" && node.data("baseLabel") === "flick") {
    return false;
  }
  return state.activeLabels.has(node.data("label")) && node.data("search").includes(state.query);
}

function applyFlickModeToGraph() {
  state.cy.batch(() => {
    state.cy.nodes().forEach((ele) => {
      const node = state.nodes[Number(ele.id().slice(1))];
      ele.data("label", effectiveLabelFor(node));
    });
  });
}

function applyFilters() {
  const cy = state.cy;
  const visibleNodes = new Set();

  cy.batch(() => {
    cy.nodes().forEach((node) => {
      const visible = nodeMatches(node);
      node.toggleClass("hidden", !visible);
      if (visible) {
        visibleNodes.add(node.id());
      }
    });

    cy.edges().forEach((edge) => {
      const visible = (
        visibleNodes.has(edge.source().id())
        && visibleNodes.has(edge.target().id())
        && edge.data("similarity") >= state.edgeThreshold
      );
      edge.toggleClass("hidden", !visible);
    });
  });

  if (state.selectedId && !visibleNodes.has(state.selectedId)) {
    clearSelection();
  }

  updateMeta();
  updateGroupCounts();
}

function emphasizeNeighborhood(node) {
  const cy = state.cy;
  const neighborhood = node.closedNeighborhood();
  cy.elements().addClass("dimmed").removeClass("neighbor");
  neighborhood.removeClass("dimmed").addClass("neighbor");
  node.removeClass("dimmed").addClass("neighbor");
}

function showSelection(ele) {
  const node = state.nodes[Number(ele.id().slice(1))];
  state.selectedNode = node;
  renderPlayViewer(node);
  const chips = displayTagsFor(node).map((tag) => (
    `<span class="chip" style="background:${colors[tag] || colors.other}">${escapeHtml(labelText[tag] || tag)}</span>`
  )).join("");
  const video = node.video_url
    ? `<a href="${escapeHtml(node.video_url)}" target="_blank" rel="noreferrer">open POV video</a>`
    : '<span class="empty-state">none</span>';
  els.selection.innerHTML = `
    <div class="play-title">${escapeHtml(node.play_id)}</div>
    <div class="chips">${chips}</div>
    <div class="kv">
      <div>Map</div><div>${escapeHtml(node.map_name)}</div>
      <div>Round</div><div>${escapeHtml(node.round)}</div>
      <div>Player</div><div>${escapeHtml(node.player_slot)}</div>
      <div>Weapon</div><div>${escapeHtml(node.weapon)}</div>
      <div>Time</div><div>${numberValue(node.event_seconds).toFixed(3)}s</div>
      <div>Distance</div><div>${numberValue(node.distance).toFixed(2)}</div>
      <div>Round kills</div><div>${escapeHtml(node.player_round_kills)}</div>
      <div>10s kills</div><div>${escapeHtml(node.kills_within_10s)}</div>
      <div>Flick score</div><div>${numberValue(node.view_snap_score).toFixed(3)}</div>
      <div>Tick data</div><div>${node.tick_feature_computed ? "yes" : "no"}</div>
      <div>Video</div><div>${video}</div>
    </div>
  `;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function clipWindowFor(node) {
  const rawDuration = numberValue(node.duration_s, 0);
  const rawEventTime = numberValue(node.event_seconds, 0);
  const duration = rawDuration > 0 ? rawDuration : rawEventTime + 3;
  const eventTime = clamp(rawEventTime, 0, duration);
  const streakStart = clamp(numberValue(node.first_player_kill_s, eventTime), 0, duration);
  const streakEnd = clamp(numberValue(node.last_player_kill_s, eventTime), 0, duration);
  const streakSpan = Math.max(0, streakEnd - streakStart);
  const isAce = node.primary_label === "ace" || node.tags.includes("ace");
  const isImpressive = node.primary_label === "impressive_multikill" || node.tags.includes("impressive_multikill");
  const isStreakClip = numberValue(node.player_round_kills) >= 3
    && streakSpan >= 0.4
    && (isAce || (isImpressive && streakSpan <= 30));

  let start = isStreakClip ? streakStart - 4 : eventTime - 4;
  let end = isStreakClip ? streakEnd + 3 : eventTime + 2.5;
  start = clamp(start, 0, duration);
  end = clamp(end, start + 0.75, duration);

  if (end - start < 5.5) {
    start = clamp(eventTime - 4, 0, duration);
    end = clamp(Math.max(eventTime + 2.5, start + 5.5), start + 0.75, duration);
  }

  return {
    start,
    end,
    eventTime,
    duration,
    mode: isStreakClip ? (isAce ? "ace" : "streak") : "kill",
  };
}

function clipUrlFor(url, clip) {
  return `${url}#t=${clip.start.toFixed(3)},${clip.end.toFixed(3)}`;
}

function renderPlayViewer(node) {
  if (!node.video_url) {
    els.playViewer.classList.add("empty-state");
    els.playViewer.innerHTML = "No POV video URL is available for this play.";
    return;
  }

  const clip = clipWindowFor(node);
  const clipUrl = clipUrlFor(node.video_url, clip);
  els.playViewer.classList.remove("empty-state");
  els.playViewer.innerHTML = `
    <div class="video-shell">
      <video id="play-video" controls playsinline preload="metadata">
        <source src="${escapeHtml(clipUrl)}" type="video/mp4">
      </video>
    </div>
    <div class="viewer-controls">
      <button type="button" data-video-step="-2">-2s</button>
      <button type="button" data-video-cue="${clip.start}">Start</button>
      <button type="button" data-video-cue="${clip.eventTime}">Kill</button>
      <button type="button" data-video-step="2">+2s</button>
      <a href="${escapeHtml(node.video_url)}" target="_blank" rel="noreferrer">source</a>
    </div>
    <div class="viewer-note">
      ${clip.mode === "ace" ? "Ace clip" : clip.mode === "streak" ? "Streak clip" : "Kill clip"}:
      ${clip.start.toFixed(1)}s-${clip.end.toFixed(1)}s, kill ${clip.eventTime.toFixed(1)}s
    </div>
  `;

  const video = byId("play-video");
  const seek = (time) => {
    if (!video) return;
    video.currentTime = clamp(time, clip.start, clip.end);
  };

  video.addEventListener("loadedmetadata", () => {
    seek(clip.start);
  }, { once: true });

  video.addEventListener("timeupdate", () => {
    if (video.currentTime >= clip.end) {
      video.pause();
      video.currentTime = clip.end;
    }
  });

  video.addEventListener("seeking", () => {
    if (video.currentTime < clip.start) {
      video.currentTime = clip.start;
    } else if (video.currentTime > clip.end) {
      video.currentTime = clip.end;
    }
  });

  video.addEventListener("error", () => {
    els.playViewer.insertAdjacentHTML(
      "beforeend",
      '<div class="viewer-error">The browser could not load this MP4 directly. Use the source link.</div>',
    );
  }, { once: true });

  els.playViewer.querySelectorAll("[data-video-cue]").forEach((button) => {
    button.addEventListener("click", () => {
      seek(Number(button.dataset.videoCue));
      video.focus();
    });
  });

  els.playViewer.querySelectorAll("[data-video-step]").forEach((button) => {
    button.addEventListener("click", () => {
      seek(video.currentTime + Number(button.dataset.videoStep));
      video.focus();
    });
  });
}

function clearSelection() {
  state.selectedId = null;
  state.selectedNode = null;
  state.cy.elements().removeClass("dimmed neighbor");
  els.selection.innerHTML = '<div class="empty-state">Select a play.</div>';
  els.playViewer.classList.add("empty-state");
  els.playViewer.innerHTML = "Select a play to load the POV video.";
}

function renderLegend() {
  const modeLabels = labelsForMode();
  els.legend.innerHTML = modeLabels.map((label) => `
    <button class="legend-button" type="button" data-label="${label}">
      <span class="swatch" style="background:${colors[label]}"></span>
      <span>${labelText[label]}</span>
    </button>
  `).join("");

  els.legend.querySelectorAll(".legend-button").forEach((button) => {
    button.addEventListener("click", () => {
      const label = button.dataset.label;
      if (state.activeLabels.has(label)) {
        state.activeLabels.delete(label);
      } else {
        state.activeLabels.add(label);
      }
      button.classList.toggle("off", !state.activeLabels.has(label));
      applyFilters();
    });
  });
}

function updateMeta() {
  const visibleNodes = state.cy.nodes(":visible").length;
  const visibleEdges = state.cy.edges(":visible").length;
  els.meta.textContent = `${visibleNodes.toLocaleString()} plays, ${visibleEdges.toLocaleString()} edges shown`;
}

function updateGroupCounts() {
  const modeLabels = labelsForMode();
  const counts = Object.fromEntries(modeLabels.map((label) => [label, 0]));
  state.cy.nodes(":visible").forEach((node) => {
    counts[node.data("label")] += 1;
  });
  els.groupCounts.innerHTML = modeLabels.map((label) => `
    <div class="count-row">
      <span class="swatch" style="background:${colors[label]}"></span>
      <span>${labelText[label]}</span>
      <strong>${counts[label].toLocaleString()}</strong>
    </div>
  `).join("");
}

function renderRunInfo() {
  const summary = state.summary;
  els.runInfo.innerHTML = `
    <div>Dataset</div><div><a href="${escapeHtml(summary.dataset_url)}" target="_blank" rel="noreferrer">Hugging Face</a></div>
    <div>Generated</div><div>${escapeHtml(summary.generated_at)}</div>
    <div>Scanned</div><div>${Number(summary.candidate_rows).toLocaleString()}</div>
    <div>Sampled</div><div>${Number(summary.sampled_rows).toLocaleString()}</div>
    <div>Edges</div><div>${Number(summary.edge_rows).toLocaleString()}</div>
    <div>Tick features</div><div>${Number(summary.tick_features_computed).toLocaleString()}</div>
    <div>Features</div><div>${Number(summary.feature_count).toLocaleString()}</div>
    <div>Flick mode</div><div>${escapeHtml(state.flickMode)}</div>
  `;
}

function attachControls() {
  els.search.addEventListener("input", () => {
    state.query = els.search.value.trim().toLowerCase();
    applyFilters();
  });

  els.flickMode.addEventListener("change", () => {
    state.flickMode = els.flickMode.value;
    state.activeLabels = new Set(labelsForMode());
    applyFlickModeToGraph();
    renderLegend();
    renderRunInfo();
    applyFilters();
    if (state.selectedId) {
      showSelection(state.cy.getElementById(state.selectedId));
    }
  });

  els.edgeStrength.addEventListener("input", () => {
    state.edgeThreshold = Number(els.edgeStrength.value) / 100;
    applyFilters();
  });

  els.layout.addEventListener("change", () => {
    state.layout = els.layout.value;
    runLayout(state.layout);
  });

  els.fit.addEventListener("click", () => {
    state.cy.fit(state.cy.elements(":visible"), 42);
  });

  els.reset.addEventListener("click", () => {
    runLayout(state.layout);
    state.cy.elements().removeClass("dimmed neighbor");
  });

  els.exportPng.addEventListener("click", () => {
    const png = state.cy.png({ full: true, scale: 2, bg: "#fcfcfb" });
    const link = document.createElement("a");
    link.href = png;
    link.download = "opencs2-play-graph.png";
    link.click();
  });
}

async function boot() {
  try {
    await loadData();
    state.activeLabels = new Set(labelsForMode());
    renderLegend();
    renderRunInfo();
    makeCy();
    attachControls();
    applyFilters();
  } catch (error) {
    console.error(error);
    els.meta.textContent = "Could not load graph data";
    byId("cy").innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

boot();
