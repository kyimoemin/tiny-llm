// ui.ts: wires the model in tiny-llm.ts to index.html. No model logic here.
import {
  BOS,
  chars,
  cleanPrefix,
  currentStep,
  docs,
  nextTokenProbs,
  params,
  reset,
  sample,
  steps,
  trainStep,
  vocab,
} from "./tiny-llm";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const trainBtn = $<HTMLButtonElement>("train");
const resetBtn = $<HTMLButtonElement>("reset");
const genBtn = $<HTMLButtonElement>("gen");
const prefixIn = $<HTMLInputElement>("prefix");
const tempIn = $<HTMLInputElement>("temp");
const canvas = $<HTMLCanvasElement>("chart");

$("info").textContent = `${params.length.toLocaleString()} parameters · ${vocab} tokens (a–z + END) · trained on ${docs.length} names`;
$("total").textContent = String(steps);
$("vocab").textContent = String(vocab);
$("chance").textContent = Math.log(vocab).toFixed(2);

// ---- training loop, time-sliced so the page stays responsive ----
let losses: number[] = [];
let avgs: number[] = [];
let running = false;

function trainChunk() {
  if (!running) return;
  const t0 = performance.now();
  const s0 = currentStep();
  // Train for ~12 ms, then give the browser a frame to repaint.
  while (performance.now() - t0 < 12 && currentStep() < steps) {
    const loss = trainStep();
    losses.push(loss);
    const prev = avgs.length ? avgs[avgs.length - 1] : loss;
    avgs.push(prev * 0.97 + loss * 0.03); // exponential moving average
  }
  const ms = performance.now() - t0;
  $("speed").textContent = String(Math.round(((currentStep() - s0) * 1000) / ms));
  render();
  if (currentStep() >= steps) setRunning(false);
  else requestAnimationFrame(trainChunk);
}

function setRunning(on: boolean) {
  running = on;
  const done = currentStep() >= steps;
  trainBtn.textContent = on ? "Pause" : done ? "Done" : currentStep() ? "Resume" : "Train";
  trainBtn.disabled = done;
  if (on) requestAnimationFrame(trainChunk);
}

trainBtn.onclick = () => setRunning(!running);
resetBtn.onclick = () => {
  setRunning(false);
  reset();
  losses = [];
  avgs = [];
  $("samples").innerHTML = "";
  $("speed").textContent = "–";
  setRunning(false); // refresh button label
  render();
};

// ---- loss chart ----
function css(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function drawChart() {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth,
    h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const g = canvas.getContext("2d")!;
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);

  const pad = { l: 34, r: 8, t: 8, b: 20 };
  const yMax = 4.5;
  const X = (i: number) => pad.l + (i / steps) * (w - pad.l - pad.r);
  const Y = (y: number) => pad.t + (1 - Math.min(y, yMax) / yMax) * (h - pad.t - pad.b);

  // axes + gridlines
  g.font = "11px system-ui, sans-serif";
  g.fillStyle = css("--muted");
  g.strokeStyle = css("--border");
  g.lineWidth = 1;
  for (let y = 0; y <= yMax; y += 1) {
    g.beginPath();
    g.moveTo(pad.l, Y(y));
    g.lineTo(w - pad.r, Y(y));
    g.stroke();
    g.fillText(y.toFixed(0), 8, Y(y) + 4);
  }
  for (let s = 0; s <= steps; s += steps / 4) g.fillText(String(s), Math.min(X(s), w - 30), h - 4);

  const line = (ys: number[], color: string, width: number) => {
    if (!ys.length) return;
    g.strokeStyle = color;
    g.lineWidth = width;
    g.beginPath();
    ys.forEach((y, i) => (i ? g.lineTo(X(i), Y(y)) : g.moveTo(X(i), Y(y))));
    g.stroke();
  };
  line(losses, css("--accent-soft"), 1);
  line(avgs, css("--accent"), 2);
}

// ---- next-token probability bars ----
function drawBars() {
  const prefix = cleanPrefix(prefixIn.value);
  $("ctx").textContent = prefix || "(start)";
  const probs = nextTokenProbs(prefix);
  const rows = probs
    .map((p, i) => ({ p, label: i === BOS ? "END" : chars[i], end: i === BOS }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 10);
  $("bars").innerHTML = rows
    .map(
      (r) => `<span class="ch${r.end ? " end" : ""}">${r.label}</span>
      <span class="track${r.end ? " end" : ""}"><span class="fill" style="width:${(r.p * 100).toFixed(1)}%"></span></span>
      <span class="pct">${(r.p * 100).toFixed(1)}%</span>`
    )
    .join("");
}

function render() {
  $("step").textContent = String(currentStep());
  $("loss").textContent = avgs.length ? avgs[avgs.length - 1].toFixed(3) : "–";
  drawChart();
  drawBars();
}

// ---- sampling ----
genBtn.onclick = () => {
  const temp = Number(tempIn.value);
  const prefix = cleanPrefix(prefixIn.value);
  $("samples").innerHTML = Array.from({ length: 12 }, () => {
    const name = sample(temp, prefix);
    return `<li><span class="p">${prefix}</span>${name.slice(prefix.length) || "<i>(ended)</i>"}</li>`;
  }).join("");
};
tempIn.oninput = () => ($("tempVal").textContent = Number(tempIn.value).toFixed(1));
prefixIn.oninput = () => drawBars();
window.addEventListener("resize", drawChart);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);

render();
