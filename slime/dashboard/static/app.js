const colors = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#ca8a04", "#475569"];
const phaseColors = {
  initialize: "#cbd5e1",
  rollout: "#159f78",
  actor_train: "#7565d8",
  update_weights: "#e87945",
  eval: "#d8b4fe",
  finished: "#94a3b8",
};
const overlays = [
  ["sglang_num_running_reqs", "running reqs"],
  ["sglang_num_queue_reqs", "queued reqs"],
  ["sglang_token_usage", "KV cache"],
  ["sglang_cache_hit_rate", "cache hit"],
  ["sglang_gen_throughput", "gen throughput"],
];

let snapshot = null;
let loadInFlight = false;
let selectedOverlay = overlays[0][0];

const fmt = (value, digits = 1) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "-";
const pct = value => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(0)}%` : "-";
const clock = ts => new Date(ts * 1000).toLocaleTimeString();
const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
})[char]);
const laneKey = lane => `${lane.node}/gpu${lane.gpu}`;

function stat(label, value, suffix = "") {
  return `<div class="stat"><b>${value}${suffix}</b><span>${label}</span></div>`;
}

function setupCanvas(canvas, height) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, canvas.clientWidth);
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function lineChart(canvas, series, yMax = null) {
  const height = Number(canvas.dataset.height || 240);
  const { ctx, width } = setupCanvas(canvas, height);
  const pad = { left: 46, right: 14, top: 14, bottom: 28 };
  const all = series.flatMap(item => item.points);
  if (!all.length) {
    ctx.fillStyle = "#68707c";
    ctx.fillText("No samples in this window", 18, 28);
    return;
  }
  const x0 = Math.min(...all.map(point => point[0]));
  const x1 = Math.max(...all.map(point => point[0]));
  const maxValue = yMax ?? Math.max(1, ...all.map(point => point[1]));
  const x = ts => pad.left + ((ts - x0) / Math.max(1, x1 - x0)) * (width - pad.left - pad.right);
  const y = value => height - pad.bottom - (value / maxValue) * (height - pad.top - pad.bottom);
  ctx.strokeStyle = "#d9dde3";
  ctx.fillStyle = "#68707c";
  ctx.font = "11px system-ui";
  for (let tick = 0; tick <= 4; tick++) {
    const value = maxValue * tick / 4;
    const yy = y(value);
    ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(width - pad.right, yy); ctx.stroke();
    ctx.fillText(fmt(value, 0), 6, yy + 4);
  }
  ctx.fillText(clock(x0), pad.left, height - 8);
  ctx.fillText(clock(x1), Math.max(pad.left, width - pad.right - 68), height - 8);
  series.forEach((item, index) => {
    ctx.strokeStyle = item.color || colors[index % colors.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    item.points.forEach((point, i) => i ? ctx.lineTo(x(point[0]), y(point[1])) : ctx.moveTo(x(point[0]), y(point[1])));
    ctx.stroke();
  });
}

function legend(element, series) {
  element.innerHTML = series.map((item, index) =>
    `<span><i style="background:${item.color || colors[index % colors.length]}"></i>${escapeHtml(item.name)}</span>`
  ).join("");
}

function gpuLanes(records) {
  const lanes = new Map();
  records.forEach(record => lanes.set(`${record.node}/gpu${record.gpu}`, { node: String(record.node), gpu: Number(record.gpu) }));
  return [...lanes.values()].sort((a, b) => a.node.localeCompare(b.node) || a.gpu - b.gpu);
}

function addressParts(address) {
  try {
    const url = new URL(address.includes("://") ? address : `http://${address}`);
    return { host: url.hostname, port: Number(url.port || 0) };
  } catch {
    return { host: String(address).split(":")[0], port: 0 };
  }
}

function mapEnginesToLanes(lanes, engineRecords) {
  const workers = [...new Set(engineRecords.map(record => record.worker_addr).filter(addr => addr && addr !== "aggregate"))];
  const mapping = new Map();
  const workerGpus = new Map();
  engineRecords.forEach(record => {
    const gpu = Number(record.labels?.gpu);
    if (record.worker_addr && Number.isInteger(gpu)) workerGpus.set(record.worker_addr, gpu);
  });
  const nodes = [...new Set(lanes.map(lane => lane.node))];
  nodes.forEach(node => {
    const nodeLanes = lanes.filter(lane => lane.node === node).sort((a, b) => a.gpu - b.gpu);
    const nodeWorkers = workers.filter(worker => addressParts(worker).host === node)
      .sort((a, b) => addressParts(a).port - addressParts(b).port);
    nodeWorkers.forEach(worker => {
      const gpu = workerGpus.get(worker);
      if (gpu !== undefined && nodeLanes.some(lane => lane.gpu === gpu)) {
        mapping.set(`${node}/gpu${gpu}`, worker);
      }
    });
    const remainingWorkers = nodeWorkers.filter(worker => ![...mapping.values()].includes(worker));
    nodeLanes.filter(lane => !mapping.has(laneKey(lane))).forEach((lane, index) => {
      if (remainingWorkers[index]) mapping.set(laneKey(lane), remainingWorkers[index]);
    });
  });
  if (!mapping.size && workers.length === lanes.length) {
    workers.sort((a, b) => addressParts(a).host.localeCompare(addressParts(b).host) || addressParts(a).port - addressParts(b).port);
    lanes.forEach((lane, index) => mapping.set(laneKey(lane), workers[index]));
  }
  return mapping;
}

function phaseIntervals(metricRecords, since, now) {
  const markers = metricRecords
    .filter(record => typeof record.metrics?.["dashboard/phase"] === "string")
    .sort((a, b) => a.ts - b.ts);
  return markers.map((record, index) => ({
    name: record.metrics["dashboard/phase"],
    t0: Math.max(since, record.ts),
    t1: Math.min(now, markers[index + 1]?.ts ?? now),
  })).filter(item => item.t1 > item.t0);
}

function drawFleet(data, lanes, engineMap) {
  const canvas = document.getElementById("fleetChart");
  const rowHeight = 66;
  const top = 30;
  const left = 132;
  const right = 16;
  const height = Math.max(150, top + lanes.length * rowHeight + 12);
  const { ctx, width } = setupCanvas(canvas, height);
  const x0 = data.since;
  const x1 = Math.max(data.now, x0 + 1);
  const plotWidth = Math.max(1, width - left - right);
  const x = ts => left + ((ts - x0) / (x1 - x0)) * plotWidth;
  const phases = phaseIntervals(data.metrics, x0, x1);
  const ratioMetric = selectedOverlay === "sglang_token_usage" || selectedOverlay === "sglang_cache_hit_rate";
  const overlayValues = data.engine.filter(record => record.metric === selectedOverlay).map(record => Number(record.value));
  const overlayMax = ratioMetric ? 1 : Math.max(1, ...overlayValues);

  ctx.font = "11px ui-monospace, SFMono-Regular, monospace";
  ctx.fillStyle = "#68707c";
  ctx.strokeStyle = "#e5e7eb";
  for (let tick = 0; tick <= 5; tick++) {
    const ts = x0 + (x1 - x0) * tick / 5;
    const xx = x(ts);
    ctx.beginPath(); ctx.moveTo(xx, 20); ctx.lineTo(xx, height - 8); ctx.stroke();
    ctx.fillText(clock(ts), Math.min(xx + 3, width - 72), 14);
  }

  lanes.forEach((lane, laneIndex) => {
    const key = laneKey(lane);
    const y0 = top + laneIndex * rowHeight;
    const shortNode = lane.node.length > 15 ? lane.node.slice(-15) : lane.node;
    ctx.fillStyle = "#20242a";
    ctx.fillText(`${shortNode} gpu ${lane.gpu}`, 8, y0 + 13);

    ctx.fillStyle = "#eef1f5";
    ctx.fillRect(left, y0, plotWidth, 12);
    phases.forEach(phase => {
      ctx.fillStyle = phaseColors[phase.name] || "#94a3b8";
      ctx.fillRect(x(phase.t0), y0, Math.max(1, x(phase.t1) - x(phase.t0)), 12);
    });

    const gpuPoints = data.gpu.filter(record => `${record.node}/gpu${record.gpu}` === key).sort((a, b) => a.ts - b.ts);
    const utilTop = y0 + 18;
    const utilHeight = 40;
    const utilY = value => utilTop + utilHeight - Math.max(0, Math.min(100, value)) / 100 * utilHeight;
    if (gpuPoints.length) {
      ctx.beginPath();
      ctx.moveTo(x(gpuPoints[0].ts), utilTop + utilHeight);
      gpuPoints.forEach(point => ctx.lineTo(x(point.ts), utilY(point.util)));
      ctx.lineTo(x(gpuPoints.at(-1).ts), utilTop + utilHeight);
      ctx.closePath();
      ctx.fillStyle = "rgba(37, 99, 235, 0.13)";
      ctx.fill();
      ctx.beginPath();
      gpuPoints.forEach((point, index) => index ? ctx.lineTo(x(point.ts), utilY(point.util)) : ctx.moveTo(x(point.ts), utilY(point.util)));
      ctx.strokeStyle = "#2563eb";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    const worker = engineMap.get(key);
    const enginePoints = data.engine.filter(record => record.worker_addr === worker && record.metric === selectedOverlay)
      .sort((a, b) => a.ts - b.ts);
    if (enginePoints.length) {
      const engineY = value => utilTop + utilHeight - Math.max(0, Number(value)) / overlayMax * utilHeight;
      ctx.beginPath();
      enginePoints.forEach((point, index) => index ? ctx.lineTo(x(point.ts), engineY(point.value)) : ctx.moveTo(x(point.ts), engineY(point.value)));
      ctx.strokeStyle = "#d97706";
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }
    ctx.strokeStyle = "#d9dde3";
    ctx.beginPath(); ctx.moveTo(left, y0 + rowHeight - 4); ctx.lineTo(width - right, y0 + rowHeight - 4); ctx.stroke();
  });

  const overlayLabel = overlays.find(([metric]) => metric === selectedOverlay)?.[1] || selectedOverlay;
  document.getElementById("overlayControls").innerHTML = [
    `<span>overlay</span>`,
    ...overlays.map(([metric, label]) => `<button type="button" data-overlay="${metric}" class="${metric === selectedOverlay ? "active" : ""}">${label}</button>`),
    `<span style="color:#b45309">${escapeHtml(overlayLabel)} scale 0–${ratioMetric ? "100%" : fmt(overlayMax, 0)}</span>`,
    `<span style="color:#68707c">live · redraws every 5s</span>`,
  ].join("");
  document.querySelectorAll("[data-overlay]").forEach(button => button.addEventListener("click", () => {
    selectedOverlay = button.dataset.overlay;
    render(snapshot);
  }));

  const seenPhases = [...new Set(phases.map(phase => phase.name))];
  document.getElementById("fleetLegend").innerHTML = [
    ...seenPhases.map(name => `<span><i style="background:${phaseColors[name] || "#94a3b8"}"></i>${escapeHtml(name)}</span>`),
    `<span><i style="background:#2563eb"></i>GPU util</span>`,
    `<span><i style="background:#d97706"></i>${escapeHtml(overlayLabel)}</span>`,
  ].join("");
}

function latestGpuByLane(records) {
  const latest = new Map();
  records.forEach(record => latest.set(`${record.node}/gpu${record.gpu}`, record));
  return latest;
}

function latestEngineValues(records) {
  const latest = new Map();
  records.forEach(record => latest.set(`${record.worker_addr}:${record.metric}`, Number(record.value)));
  return latest;
}

function renderGpuCards(data, lanes, engineMap) {
  const gpuLatest = latestGpuByLane(data.gpu);
  const engineLatest = latestEngineValues(data.engine);
  document.getElementById("gpuCards").innerHTML = lanes.map(lane => {
    const key = laneKey(lane);
    const gpu = gpuLatest.get(key) || {};
    const worker = engineMap.get(key);
    const value = metric => engineLatest.get(`${worker}:${metric}`);
    const memoryRatio = gpu.mem_total_mb ? gpu.mem_used_mb / gpu.mem_total_mb : NaN;
    const workerLabel = worker ? worker.replace(/^https?:\/\//, "") : "engine unavailable";
    return `<div class="gpu-card">
      <div class="gpu-card-head"><b>${escapeHtml(lane.node)} · GPU ${lane.gpu}</b><span title="${escapeHtml(workerLabel)}">${escapeHtml(workerLabel)}</span></div>
      <div class="gpu-card-values">
        <div><b>${fmt(gpu.util, 0)}%</b><span>GPU util</span></div>
        <div><b>${pct(memoryRatio)}</b><span>VRAM</span></div>
        <div><b>${pct(value("sglang_token_usage"))}</b><span>KV cache</span></div>
        <div><b>${fmt(value("sglang_num_running_reqs"), 0)}</b><span>run / q ${fmt(value("sglang_num_queue_reqs"), 0)}</span></div>
      </div>
      <div class="bar"><i style="width:${Math.max(0, Math.min(100, Number(gpu.util) || 0))}%"></i></div>
    </div>`;
  }).join("");
}

function requestSeries(records) {
  const wanted = new Map([
    ["sglang_num_running_reqs", "running requests"],
    ["sglang_num_queue_reqs", "queued requests"],
    ["sglang_num_prefill_inflight_queue_reqs", "prefill inflight"],
    ["sglang_num_decode_transfer_queue_reqs", "decode transfer queue"],
  ]);
  const grouped = new Map();
  records.filter(record => wanted.has(record.metric)).forEach(record => {
    const bucket = `${record.metric}:${record.ts}`;
    grouped.set(bucket, (grouped.get(bucket) || 0) + Number(record.value));
  });
  return [...wanted].map(([metric, name], index) => ({
    name,
    color: colors[index],
    points: [...grouped].filter(([key]) => key.startsWith(`${metric}:`))
      .map(([key, value]) => [Number(key.slice(metric.length + 1)), value]).sort((a, b) => a[0] - b[0]),
  })).filter(item => item.points.length);
}

function buildSpans(events, now = Date.now() / 1000) {
  const taskByTrace = new Map();
  const starts = new Map();
  const spans = [];
  events.forEach(event => {
    if (event.name === "fully_async_dispatch" && event.attrs?.task_type) taskByTrace.set(event.trace_id, event.attrs.task_type);
    if (event.type === "span_start") starts.set(event.span_id, event);
    if (event.type === "span_end" && starts.has(event.span_id)) {
      const start = starts.get(event.span_id);
      starts.delete(event.span_id);
      const attrs = { ...(start.attrs || {}), ...(event.attrs || {}) };
      if (!attrs.task_type && taskByTrace.has(event.trace_id)) attrs.task_type = taskByTrace.get(event.trace_id);
      spans.push({
        ts: start.ts, end: event.ts, sample: event.sample_id, group: event.group_id,
        name: event.name, duration: event.ts - start.ts, attrs,
      });
    }
  });
  starts.forEach(start => {
    if (!["generation_turn", "tool_call", "reward_model"].includes(start.name)) return;
    spans.push({
      ts: start.ts, end: now, sample: start.sample_id, group: start.group_id,
      name: start.name, duration: Math.max(0, now - start.ts), attrs: start.attrs || {}, ongoing: true,
    });
  });
  return spans.filter(span => ["generation_turn", "tool_call", "reward_model"].includes(span.name));
}

function toolConcurrencySeries(spans, compactSeries = null) {
  const domains = [["math", "Math tools", "#d97706"], ["qa", "QA tools", "#059669"]];
  return domains.map(([domain, name, color]) => {
    if (compactSeries?.[domain]) return { name, color, points: compactSeries[domain] };
    const changes = [];
    spans.filter(span => span.name === "tool_call" && (span.attrs.is_tool_call === true || span.ongoing) && span.attrs.task_type === domain)
      .forEach(span => { changes.push([span.ts, 1]); changes.push([span.end, -1]); });
    changes.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    let running = 0;
    const points = [];
    changes.forEach(([ts, delta]) => { points.push([ts, running]); running += delta; points.push([ts, running]); });
    return { name, color, points };
  }).filter(item => item.points.length);
}

function renderTraces(data) {
  const summary = data.trace_summary;
  const spans = summary?.spans || buildSpans(data.trace, data.now);
  const generations = spans.filter(item => item.name === "generation_turn");
  const actions = spans.filter(item => item.name === "tool_call");
  const tools = actions.filter(item => item.attrs.is_tool_call === true || item.ongoing);
  const toolCallCount = summary?.totals?.tool_calls
    ?? tools.reduce((total, item) => total + Number(item.attrs.tool_calls || 1), 0);
  const average = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
  document.getElementById("traceSummary").innerHTML = [
    stat("generation turns", summary?.totals?.generation_turns ?? generations.length),
    stat("mean generation", fmt(summary?.totals?.mean_generation_seconds ?? average(generations.map(item => item.duration)), 2), "s"),
    stat("tool calls", toolCallCount),
    stat("mean tool call", fmt(summary?.totals?.mean_tool_seconds ?? average(tools.map(item => item.duration)), 2), "s"),
    stat("max tool call", fmt(summary?.totals?.max_tool_seconds ?? Math.max(0, ...tools.map(item => item.duration)), 2), "s"),
  ].join("");
  document.getElementById("traceRows").innerHTML = spans.sort((a, b) => b.ts - a.ts).slice(0, 500).map(item => {
    const details = Object.entries(item.attrs).filter(([key]) => !["task_type", "turn"].includes(key))
      .map(([key, value]) => `${key}=${value}`);
    if (item.ongoing) details.unshift("ongoing=true");
    const detail = details.join(" ");
    return `<tr><td>${escapeHtml(clock(item.ts))}</td><td>${escapeHtml(item.sample ?? "-")}</td><td>${escapeHtml(item.attrs.task_type ?? "-")}</td><td>${escapeHtml(item.attrs.turn ?? "-")}</td><td>${escapeHtml(item.name)}</td><td>${fmt(item.duration, 3)}s</td><td>${escapeHtml(detail)}</td></tr>`;
  }).join("");
  return { spans, generations, tools, toolCallCount };
}

function render(data) {
  snapshot = data;
  const lanes = gpuLanes(data.gpu);
  const engineMap = mapEnginesToLanes(lanes, data.engine);
  const requests = requestSeries(data.engine);
  const trace = renderTraces(data);
  const tools = toolConcurrencySeries(trace.spans, data.trace_summary?.tool_series);
  const gpuLatest = [...latestGpuByLane(data.gpu).values()];
  const engineLatest = latestEngineValues(data.engine);
  const workers = [...new Set(engineMap.values())];
  const avgGpu = gpuLatest.length ? gpuLatest.reduce((sum, row) => sum + Number(row.util), 0) / gpuLatest.length : 0;
  const kvValues = workers.map(worker => engineLatest.get(`${worker}:sglang_token_usage`)).filter(Number.isFinite);
  const avgKv = kvValues.length ? kvValues.reduce((a, b) => a + b, 0) / kvValues.length : NaN;
  const latestRunning = requests.find(item => item.name === "running requests")?.points.at(-1)?.[1] ?? 0;
  const latestQueued = requests.find(item => item.name === "queued requests")?.points.at(-1)?.[1] ?? 0;
  const activeTools = tools.reduce((sum, item) => sum + Number(item.points.at(-1)?.[1] || 0), 0);
  document.getElementById("summary").innerHTML = [
    stat("current mean GPU", fmt(avgGpu), "%"),
    stat("GPU devices", lanes.length),
    stat("mean KV cache", pct(avgKv)),
    stat("running requests", fmt(latestRunning, 0)),
    stat("queued requests", fmt(latestQueued, 0)),
    stat("active / total tools", `${activeTools} / ${trace.toolCallCount}`),
  ].join("");
  const meta = data.meta;
  const nodeCount = new Set(lanes.map(lane => lane.node)).size;
  document.getElementById("runMeta").textContent = `${meta.run_name || "slime-run"} · ${nodeCount} nodes · ${lanes.length} GPUs · ${workers.length} engines · schema ${meta.schema_version ?? "?"}`;
  drawFleet(data, lanes, engineMap);
  renderGpuCards(data, lanes, engineMap);
  lineChart(document.getElementById("requestChart"), requests);
  legend(document.getElementById("requestLegend"), requests);
  lineChart(document.getElementById("toolChart"), tools);
  legend(document.getElementById("toolLegend"), tools);
  document.getElementById("updated").textContent = `updated ${new Date().toLocaleTimeString()}`;
}

async function load() {
  if (loadInFlight) return;
  loadInFlight = true;
  try {
    const minutes = document.getElementById("windowMinutes").value;
    const response = await fetch(`/api/snapshot?minutes=${minutes}&raw_engine=true`, { cache: "no-store" });
    if (!response.ok) throw new Error(`snapshot request failed: ${response.status}`);
    render(await response.json());
  } finally {
    loadInFlight = false;
  }
}

document.getElementById("refresh").addEventListener("click", () => load().catch(console.error));
document.getElementById("windowMinutes").addEventListener("change", () => load().catch(console.error));
window.addEventListener("resize", () => snapshot && render(snapshot));
load().catch(error => { document.getElementById("updated").textContent = error.message; });
setInterval(() => load().catch(console.error), 5000);
