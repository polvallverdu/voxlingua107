// Runs in a real browser (see run.ts). Loads the model like an app would: fetched over HTTP,
// verified with Web Crypto, cached in Cache Storage.
import { LanguageIdentifier, decodeWav } from "../../src/web.ts";

declare global {
  interface Window {
    result?: unknown;
  }
}

try {
  const t0 = performance.now();
  const lid = await LanguageIdentifier.load({ urls: [new URL("/model.onnx", location.href).href] });
  const loadMs = performance.now() - t0;

  const { samples, sampleRate } = decodeWav(await (await fetch("/udhr_th.wav")).arrayBuffer());
  const t1 = performance.now();
  const result = await lid.identify(samples, { sampleRate, topK: 3 });
  window.result = { ok: true, result, loadMs, identifyMs: performance.now() - t1 };
} catch (err) {
  window.result = { ok: false, error: String(err instanceof Error ? (err.stack ?? err.message) : err) };
}
