// Imported from languages.ts (not web.ts) so the page bundle doesn't include onnxruntime-web.
import { LANGUAGES, isLanguageCode, type LanguageCode } from "../src/languages.ts";
import type { Prediction } from "../src/web.ts";
import type { WorkerRequest, WorkerResponse } from "./worker.ts";

// Live mode: analyze the last WINDOW_S seconds of mic audio, as often as the model keeps up.
const WINDOW_S = 6;
const MIN_S = 1.5;
const SPEECH_RMS = 0.012; // below this the last second counts as silence
const SMOOTHING = 0.55; // weight of the newest result in the displayed (smoothed) ranking

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
const pickerSummaryEl = $("langs-summary");
const pickerListEl = $("lang-list");
const pickerSearchEl = $<HTMLInputElement>("lang-search");
const chipsEl = $("lang-chips");

// --- Worker ----------------------------------------------------------------------------------

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
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
let selectionVersion = 0; // bumped on change so live mode resets its smoothing

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
  selectionVersion++;
  renderSelection();
  if (!mic && lastClip) void runClip(lastClip);
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

// --- Microphone ------------------------------------------------------------------------------

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

let mic: { ctx: AudioContext; stream: MediaStream } | undefined;

micBtn.onclick = async () => {
  if (mic) return stopMic();
  micBtn.disabled = true;
  try {
    // Ask for the mic while the model downloads.
    const [stream] = await Promise.all([
      navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: true },
      }),
      loadModel(),
    ]);
    await startMic(stream);
  } catch (err) {
    setStatus(`Couldn't start: ${(err as Error).message}`);
  } finally {
    micBtn.disabled = false;
  }
};

async function startMic(stream: MediaStream) {
  // Use the device's native rate (Firefox can't mix rates); the library resamples to 16 kHz.
  const ctx = new AudioContext();
  const rate = ctx.sampleRate;
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([TAP_WORKLET], { type: "text/javascript" })));
  const tap = new AudioWorkletNode(ctx, "tap");
  const mute = new GainNode(ctx, { gain: 0 }); // keeps the graph pulling without playing the mic back
  ctx.createMediaStreamSource(stream).connect(tap).connect(mute).connect(ctx.destination);
  mic = { ctx, stream };

  const ring = new Float32Array(rate * WINDOW_S);
  let write = 0;
  let filled = 0;
  let level = 0;
  tap.port.onmessage = ({ data }: MessageEvent<Float32Array>) => {
    let sum = 0;
    for (const s of data) {
      ring[write] = s;
      write = (write + 1) % ring.length;
      sum += s * s;
    }
    filled = Math.min(ring.length, filled + data.length);
    level = Math.max(Math.sqrt(sum / data.length), level * 0.85);
  };

  const latest = (n: number) => {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = ring[(write - n + i + ring.length) % ring.length]!;
    return out;
  };
  micBtn.textContent = "Stop microphone";
  micBtn.classList.add("live");
  setStatus("Listening… speak in any language.");

  const meter = () => {
    if (mic?.ctx !== ctx) return;
    levelEl.style.width = `${Math.min(100, level * 400)}%`;
    requestAnimationFrame(meter);
  };
  meter();

  void listen(ctx, rate, latest, () => filled);
}

async function listen(
  ctx: AudioContext,
  rate: number,
  latest: (n: number) => Float32Array,
  filled: () => number,
) {
  const rms = (x: Float32Array) => Math.sqrt(x.reduce((a, s) => a + s * s, 0) / x.length);
  const smoothed = new Map<string, Prediction>();
  let version = selectionVersion;
  while (mic?.ctx === ctx) {
    await new Promise((r) => setTimeout(r, 250));
    if (filled() < rate * MIN_S) continue;
    if (rms(latest(rate)) < SPEECH_RMS) {
      setStatus("Listening… (silence)");
      continue;
    }
    setStatus("Listening…");
    const top = await identify(latest(filled()), rate);
    if (mic?.ctx !== ctx) break;
    if (version !== selectionVersion) {
      // Languages changed: start over instead of mixing old and new rankings.
      version = selectionVersion;
      smoothed.clear();
    }

    // Exponential moving average over results so the display doesn't flicker.
    for (const p of smoothed.values()) p.probability *= 1 - SMOOTHING;
    for (const p of top) {
      const prev = smoothed.get(p.code);
      smoothed.set(p.code, { ...p, probability: (prev?.probability ?? 0) + SMOOTHING * p.probability });
    }
    render([...smoothed.values()].sort((a, b) => b.probability - a.probability).slice(0, 5));
  }
}

function stopMic() {
  mic?.stream.getTracks().forEach((t) => t.stop());
  void mic?.ctx.close();
  mic = undefined;
  levelEl.style.width = "0";
  micBtn.textContent = "Start microphone";
  micBtn.classList.remove("live");
  setStatus("Stopped.");
}

// --- Files -----------------------------------------------------------------------------------

interface Clip {
  samples: Float32Array;
  sampleRate: number;
  label: string;
}
let lastClip: Clip | undefined; // re-analyzed when the language selection changes

async function runClip(clip: Clip) {
  setStatus(`Analyzing ${clip.label}…`);
  // identify() transfers the buffer to the worker, so send a copy.
  render(await identify(clip.samples.slice(), clip.sampleRate));
  const seconds = (clip.samples.length / clip.sampleRate).toFixed(1);
  setStatus(`Result for ${clip.label} (${seconds} s analyzed).`);
}

async function identifyBuffer(data: ArrayBuffer, label: string) {
  try {
    if (mic) stopMic();
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
