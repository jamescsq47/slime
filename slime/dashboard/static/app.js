const colors = ["#1677b8", "#1f8a5b", "#d97706", "#b34747", "#6b5ca5", "#008b8b", "#9a6700", "#525b66"];
let snapshot = null;
let loadInFlight = false;

const fmt = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : "-";
const clock = (ts) => new Date(ts * 1000).toLocaleTimeString();
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
})[char]);

function stat(label, value, suffix = "") {
  return `<div class="stat"><b>${value}${suffix}</b><span>${label}</span></div>`;
}

function lineChart(canvas, series, yMax = null) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, canvas.clientWidth);
  const height = Number(canvas.dataset.height || 240);
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 46, right: 14, top: 14, bottom: 28 };
  const all = series.flatMap(item => item.points);
  if (!all.length) {
    ctx.fillStyle = "#68707c";
    ctx.fillText("No samples in this window", 18, 28);
    return;
  }
  const x0 = Math.min(...all.map(p => p[0]));
  const x1 = Math.max(...all.map(p => p[0]));
  const maxValue = yMax ?? Math.max(1, ...all.map(p => p[1]));
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
  element.innerHTML = series.map((item, index) => `<span><i style="background:${item.color || colors[index % colors.length]}"></i>${item.name}</span>`).join("");
}

function gpuSeries(records) {
  const grouped = new Map();
  records.forEach(record => {
    const key = `${record.node}/gpu${record.gpu}`;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push([record.ts, record.util]);
  });
  return [...grouped].map(([name, points], index) => ({ name, points, color: colors[index % colors.length] }));
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
    points: [...grouped].filter(([key]) => key.startsWith(`${metric}:`)).map(([key, value]) => [Number(key.slice(metric.length + 1)), value]).sort((a, b) => a[0] - b[0]),
  })).filter(item => item.points.length);
}

function buildSpans(events) {
  const taskByTrace = new Map();
  const starts = new Map();
  const spans = [];
  events.forEach(event => {
    if (event.name === "fully_async_dispatch" && event.attrs?.task_type) {
      taskByTrace.set(event.trace_id, event.attrs.task_type);
    }
    if (event.type === "span_start") starts.set(event.span_id, event);
    if (event.type === "span_end" && starts.has(event.span_id)) {
      const start = starts.get(event.span_id);
      const attrs = { ...(start.attrs || {}), ...(event.attrs || {}) };
      if (!attrs.task_type && taskByTrace.has(event.trace_id)) {
        attrs.task_type = taskByTrace.get(event.trace_id);
      }
      spans.push({
        ts: start.ts,
        sample: event.sample_id,
        group: event.group_id,
        name: event.name,
        duration: event.ts - start.ts,
        attrs,
      });
    }
  });
  return spans.filter(span => ["generation_turn", "tool_call", "reward_model"].includes(span.name));
}

function renderTraces(events) {
  const spans = buildSpans(events);
  const generations = spans.filter(item => item.name === "generation_turn");
  const actions = spans.filter(item => item.name === "tool_call");
  const tools = actions.filter(item => item.attrs.is_tool_call === true);
  const toolCallCount = tools.reduce((total, item) => total + Number(item.attrs.tool_calls || 1), 0);
  const lifecycle = events.filter(event => event.type === "event");
  const countLifecycleGroups = name => new Set(lifecycle.filter(event => event.name === name).map(event =>
    `${event.attrs?.sample_id ?? event.group_id}:${event.ts}`
  )).size;
  const evicted = countLifecycleGroups("fully_async_evicted");
  const selected = countLifecycleGroups("fully_async_selected");
  const average = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
  document.getElementById("traceSummary").innerHTML = [
    stat("generation turns", generations.length),
    stat("mean generation", fmt(average(generations.map(item => item.duration)), 2), "s"),
    stat("tool calls", toolCallCount),
    stat("mean tool call", fmt(average(tools.map(item => item.duration)), 2), "s"),
    stat("selected / evicted", `${selected} / ${evicted}`),
  ].join("");
  document.getElementById("traceRows").innerHTML = spans.sort((a, b) => b.ts - a.ts).slice(0, 500).map(item => {
    const detail = Object.entries(item.attrs).filter(([key]) => !["task_type", "turn"].includes(key)).map(([key, value]) => `${key}=${value}`).join(" ");
    return `<tr><td>${escapeHtml(clock(item.ts))}</td><td>${escapeHtml(item.sample ?? "-")}</td><td>${escapeHtml(item.attrs.task_type ?? "-")}</td><td>${escapeHtml(item.attrs.turn ?? "-")}</td><td>${escapeHtml(item.name)}</td><td>${fmt(item.duration, 3)}s</td><td>${escapeHtml(detail)}</td></tr>`;
  }).join("");
  return { generations, tools, toolCallCount, evicted, selected };
}

function render(data) {
  snapshot = data;
  const gpu = gpuSeries(data.gpu);
  const requests = requestSeries(data.engine);
  const trace = renderTraces(data.trace);
  const avgGpu = data.gpu.length ? data.gpu.reduce((sum, row) => sum + row.util, 0) / data.gpu.length : 0;
  const latestRunning = requests.find(item => item.name === "running requests")?.points.at(-1)?.[1] ?? 0;
  const latestQueued = requests.find(item => item.name === "queued requests")?.points.at(-1)?.[1] ?? 0;
  document.getElementById("summary").innerHTML = [
    stat("mean GPU utilization", fmt(avgGpu), "%"),
    stat("GPU devices reporting", gpu.length),
    stat("running requests", fmt(latestRunning, 0)),
    stat("queued requests", fmt(latestQueued, 0)),
    stat("generation turns", trace.generations.length),
    stat("tool calls", trace.toolCallCount),
  ].join("");
  const meta = data.meta;
  document.getElementById("runMeta").textContent = `${meta.run_name || "slime-run"} · ${meta.args?.rollout_num_gpus ?? "?"} rollout GPUs · schema ${meta.schema_version ?? "?"}`;
  lineChart(document.getElementById("gpuChart"), gpu, 100);
  legend(document.getElementById("gpuLegend"), gpu);
  lineChart(document.getElementById("requestChart"), requests);
  legend(document.getElementById("requestLegend"), requests);
  document.getElementById("updated").textContent = `updated ${new Date().toLocaleTimeString()}`;
}

async function load() {
  if (loadInFlight) return;
  loadInFlight = true;
  try {
    const minutes = document.getElementById("windowMinutes").value;
    const response = await fetch(`/api/snapshot?minutes=${minutes}`, { cache: "no-store" });
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
