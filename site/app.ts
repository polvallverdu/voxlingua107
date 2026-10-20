// Imported from languages.ts (not web.ts) so the page bundle doesn't include onnxruntime-web.
import { LANGUAGES, isLanguageCode, type LanguageCode } from "../src/languages.ts";
import type { Prediction } from "../src/web.ts";
import type { WorkerRequest, WorkerResponse } from "./worker.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const statusEl = $("status");
const progressEl = $<HTMLProgressElement>("progress");
const micBtn = $<HTMLButtonElement>("mic");
const sampleBtn = $<HTMLButtonElement>("sample");
const fileInput = $<HTMLInputElement>("file");
const languageEl = $("language");
const confidenceEl = $("confidence");
const barsEl = $("bars");
const levelEl = $("level");
const timingEl = $("timing");
const playbackEl = $("playback");
const pickerSummaryEl = $("langs-summary");
const pickerListEl = $("lang-list");
const pickerSearchEl = $<HTMLInputElement>("lang-search");
const chipsEl = $("lang-chips");

// --- Worker ----------------------------------------------------------------------------------

// The worker's filename has a content hash (injected by scripts/site.ts), so a deploy can't pair a
// new page with a stale cached worker.
const worker = new Worker(new URL(`./${process.env.WORKER_FILE ?? "worker.js"}`, import.meta.url), { type: "module" });
const pending = new Map<number, { resolve: (r: Prediction[]) => void; reject: (e: Error) => void }>();
let nextId = 0;
let loaded: Promise<void> | undefined;
let resolveLoaded: () => void;
let rejectLoaded: (e: Error) => void;

worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
  if (data.type === "progress") {
    progressEl.hidden = false;
    progressEl.max = data.total || 1;
    progressEl.value = data.loaded;
    setStatus(`Downloading the model… ${mb(data.loaded)}${data.total ? ` / ${mb(data.total)}` : ""} MB (only the first time)`);
  } else if (data.type === "ready") {
    progressEl.hidden = true;
    resolveLoaded();
  } else if (data.type === "result") {
    timingEl.textContent = `Last analysis: ${data.ms.toFixed(0)} ms in your browser (WebAssembly).`;
    pending.get(data.id)?.resolve(data.result.top);
    pending.delete(data.id);
  } else if (data.id === undefined) {
    progressEl.hidden = true;
    loaded = undefined;
    rejectLoaded(new Error(data.message));
  } else {
    pending.get(data.id)?.reject(new Error(data.message));
    pending.delete(data.id);
  }
};

function loadModel(): Promise<void> {
  loaded ??= new Promise<void>((resolve, reject) => {
    resolveLoaded = resolve;
    rejectLoaded = reject;
    setStatus("Loading the model…");
    worker.postMessage({ type: "load" } satisfies WorkerRequest);
  });
  return loaded;
}

function identify(samples: Float32Array, sampleRate: number): Promise<Prediction[]> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    const candidates = selected.size ? [...selected] : undefined;
    worker.postMessage({ type: "identify", id, samples, sampleRate, candidates } satisfies WorkerRequest, [
      samples.buffer,
    ]);
  });
}

// --- Language picker -----------------------------------------------------------------------
// Choosing among a few expected languages (the `candidates` option) is much more accurate than
// among all 107.

const STORAGE_KEY = "voxlingua107-demo:languages";
const selected = new Set<LanguageCode>(readSelection());

function readSelection(): LanguageCode[] {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as string[];
    return saved.filter(isLanguageCode);
  } catch {
    return [];
  }
}

const byName = [...LANGUAGES].sort((a, b) => a.name.localeCompare(b.name));
const checkboxes = new Map<LanguageCode, HTMLInputElement>();
pickerListEl.replaceChildren(
  ...byName.map((lang) => {
    const label = document.createElement("label");
    label.dataset.search = `${lang.name} ${lang.code}`.toLowerCase();
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = lang.code;
    box.onchange = () => {
      if (box.checked) selected.add(lang.code);
      else selected.delete(lang.code);
      selectionChanged();
    };
    checkboxes.set(lang.code, box);
    label.append(box, ` ${lang.name} `);
    const code = document.createElement("span");
    code.className = "code";
    code.textContent = lang.code;
    label.append(code);
    return label;
  }),
);

pickerSearchEl.oninput = () => {
  const q = pickerSearchEl.value.trim().toLowerCase();
  for (const label of pickerListEl.children as HTMLCollectionOf<HTMLElement>) {
    // Match the start of any word in the name, or the code ("ser" -> Serbian, "es" -> Spanish).
    label.hidden = !!q && !label.dataset.search!.split(" ").some((word) => word.startsWith(q));
  }
};

$("lang-browser").onclick = () => {
  // "en-US" -> "en"; Norwegian Bokmål is "no" in VoxLingua107.
  const codes = navigator.languages
    .map((l) => l.split("-")[0]!.toLowerCase())
    .map((c) => (c === "nb" ? "no" : c))
    .filter(isLanguageCode);
  if (!codes.length) return setStatus("None of your browser languages are in the list; pick them by hand.");
  codes.forEach((c) => selected.add(c));
  selectionChanged();
};

$("lang-all").onclick = () => {
  selected.clear();
  selectionChanged();
};

function selectionChanged() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...selected]));
  } catch {}
  renderSelection();
  if (!recording && lastClip) void runClip(lastClip);
}

function renderSelection() {
  for (const [code, box] of checkboxes) box.checked = selected.has(code);
  const names = byName.filter((l) => selected.has(l.code));
  pickerSummaryEl.textContent =
    names.length === 0
      ? "all 107"
      : names.length === 1
        ? `${names[0]!.name} only (pick at least 2)`
        : `${names.length} selected`;
  chipsEl.replaceChildren(
    ...names.map((lang) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.title = `Remove ${lang.name}`;
      chip.textContent = `${lang.name} ×`;
      chip.onclick = () => {
        selected.delete(lang.code);
        selectionChanged();
      };
      return chip;
    }),
  );
}
renderSelection();

// --- Rendering -------------------------------------------------------------------------------

const mb = (n: number) => (n / 1e6).toFixed(0);
const setStatus = (text: string) => (statusEl.textContent = text);

function render(top: Prediction[], hint?: string) {
  const [best] = top;
  if (!best) return;
  languageEl.textContent = best.name;
  confidenceEl.textContent = hint ?? `${best.code} · ${(best.probability * 100).toFixed(0)}% confident`;
  barsEl.replaceChildren(
    ...top.map((p) => {
      const row = document.createElement("li");
      row.innerHTML = `<span class="name"></span><span class="bar"><span></span></span><span class="pct"></span>`;
      row.querySelector(".name")!.textContent = `${p.name} (${p.code})`;
      (row.querySelector(".bar > span") as HTMLElement).style.width = `${(p.probability * 100).toFixed(1)}%`;
      row.querySelector(".pct")!.textContent = `${(p.probability * 100).toFixed(1)}%`;
      return row;
    }),
  );
}

// --- Recording -------------------------------------------------------------------------------

const RECORD_S = 10;
const MIN_RECORD_S = 1;

const TAP_WORKLET = `
class Tap extends AudioWorkletProcessor {
  buf = new Float32Array(2048); n = 0;
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === this.buf.length) { this.port.postMessage(this.buf); this.buf = new Float32Array(2048); this.n = 0; }
    }
    return true;
  }
}
registerProcessor("tap", Tap);`;

let recording: { finish: () => void } | undefined;

micBtn.onclick = async () => {
  if (recording) return recording.finish();
  micBtn.disabled = true;
  try {
    // Start downloading the model while the user records.
    void loadModel().catch((err: Error) => setStatus(`Couldn't load the model: ${err.message}`));
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: true },
    });
    latestRun++; // drop results of analyses still running from before the recording
    const clip = await record(stream);
    if (clip.samples.length < clip.sampleRate * MIN_RECORD_S) {
      setStatus("That was too short. Record at least a couple of seconds of speech.");
      return;
    }
    showPlayback(clip);
    lastClip = clip;
    await loadModel();
    await runClip(clip);
  } catch (err) {
    setStatus(`Couldn't record: ${(err as Error).message}`);
  } finally {
    recording = undefined;
    micBtn.disabled = false;
    micBtn.textContent = `Record ${RECORD_S} seconds`;
    micBtn.classList.remove("live");
    levelEl.style.width = "0";
    levelEl.parentElement!.hidden = true;
  }
};

/** Record up to RECORD_S seconds from the mic. Resolves when time is up or `recording.finish()` is called. */
async function record(stream: MediaStream): Promise<Clip> {
  // Use the device's native rate (Firefox can't mix rates); the library resamples to 16 kHz.
  const ctx = new AudioContext();
  const sampleRate = ctx.sampleRate;
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([TAP_WORKLET], { type: "text/javascript" })));
  const tap = new AudioWorkletNode(ctx, "tap");
  const mute = new GainNode(ctx, { gain: 0 }); // keeps the graph pulling without playing the mic back
  ctx.createMediaStreamSource(stream).connect(tap).connect(mute).connect(ctx.destination);

  const chunks: Float32Array[] = [];
  let length = 0;
  let level = 0;
  tap.port.onmessage = ({ data }: MessageEvent<Float32Array>) => {
    chunks.push(data);
    length += data.length;
    level = Math.max(Math.sqrt(data.reduce((a, s) => a + s * s, 0) / data.length), level * 0.85);
  };

  micBtn.disabled = false;
  levelEl.parentElement!.hidden = false;
  micBtn.textContent = "Stop and analyze";
  micBtn.classList.add("live");

  await new Promise<void>((resolve) => {
    const started = performance.now();
    const tick = () => {
      if (!recording) return;
      const elapsed = (performance.now() - started) / 1000;
      if (elapsed >= RECORD_S) return recording.finish();
      setStatus(`Recording… ${Math.ceil(RECORD_S - elapsed)} s left. Speak normally.`);
      levelEl.style.width = `${Math.min(100, level * 400)}%`;
      requestAnimationFrame(tick);
    };
    recording = { finish: resolve };
    tick();
  });
  recording = undefined;

  stream.getTracks().forEach((t) => t.stop());
  await ctx.close();
  const samples = new Float32Array(length);
  let offset = 0;
  for (const c of chunks) {
    samples.set(c, offset);
    offset += c.length;
  }
  return { samples, sampleRate, label: "your recording" };
}

/** Let the user listen to what was analyzed. */
function showPlayback(clip: Clip) {
  const old = playbackEl.querySelector("audio")?.src;
  if (old) URL.revokeObjectURL(old);
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src = URL.createObjectURL(encodeWav(clip.samples, clip.sampleRate));
  playbackEl.replaceChildren(audio);
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const str = (o: number, s: string) => [...s].forEach((c, i) => view.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  str(36, "data");
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 32767, true));
  return new Blob([view.buffer], { type: "audio/wav" });
}

// --- Files -----------------------------------------------------------------------------------

interface Clip {
  samples: Float32Array;
  sampleRate: number;
  label: string;
}
let lastClip: Clip | undefined; // re-analyzed when the language selection changes

let latestRun = 0;

async function runClip(clip: Clip) {
  const run = ++latestRun;
  setStatus(`Analyzing ${clip.label}…`);
  // identify() transfers the buffer to the worker, so send a copy.
  const top = await identify(clip.samples.slice(), clip.sampleRate);
  if (run !== latestRun) return; // a newer analysis (other clip or language selection) superseded this one
  render(top);
  const seconds = (clip.samples.length / clip.sampleRate).toFixed(1);
  setStatus(`Result for ${clip.label} (${seconds} s analyzed).`);
}

async function identifyBuffer(data: ArrayBuffer, label: string) {
  try {
    await loadModel();
    setStatus(`Analyzing ${label}…`);
    // decodeAudioData handles wav, mp3, ogg, m4a, ... (whatever the browser supports).
    const audio = await new AudioContext().decodeAudioData(data);
    const seconds = Math.min(audio.duration, 60); // the first minute is plenty
    const n = Math.floor(seconds * audio.sampleRate);
    const mono = new Float32Array(n);
    for (let c = 0; c < audio.numberOfChannels; c++) {
      const ch = audio.getChannelData(c);
      for (let i = 0; i < n; i++) mono[i]! += ch[i]! / audio.numberOfChannels;
    }
    lastClip = { samples: mono, sampleRate: audio.sampleRate, label };
    await runClip(lastClip);
  } catch (err) {
    setStatus(`Couldn't analyze ${label}: ${(err as Error).message}`);
  }
}

sampleBtn.onclick = async () => {
  const res = await fetch(new URL("./udhr_th.wav", location.href));
  await identifyBuffer(await res.arrayBuffer(), "the Thai sample clip");
};

fileInput.onchange = async () => {
  const file = fileInput.files?.[0];
  if (file) await identifyBuffer(await file.arrayBuffer(), file.name);
  fileInput.value = "";
};

// Copy buttons on code blocks.
for (const pre of document.querySelectorAll("pre")) {
  const btn = document.createElement("button");
  btn.className = "copy";
  btn.textContent = "Copy";
  btn.onclick = async () => {
    await navigator.clipboard.writeText(pre.querySelector("code")?.textContent ?? "");
    btn.textContent = "Copied";
    setTimeout(() => (btn.textContent = "Copy"), 1200);
  };
  pre.append(btn);
}
