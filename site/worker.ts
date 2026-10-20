// Runs the model off the main thread so the UI (level meter, bars) stays smooth.
import { LanguageIdentifier, type IdentifyResult, type LanguageCode } from "../src/web.ts";

export type WorkerRequest =
  | { type: "load" }
  | { type: "identify"; id: number; samples: Float32Array; sampleRate: number; candidates?: LanguageCode[] };

export type WorkerResponse =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; loadMs: number }
  | { type: "result"; id: number; result: IdentifyResult; ms: number }
  | { type: "error"; id?: number; message: string };

let lid: Promise<LanguageIdentifier> | undefined;
const post = (msg: WorkerResponse) => postMessage(msg);

onmessage = async ({ data }: MessageEvent<WorkerRequest>) => {
  if (data.type === "load") {
    const t0 = performance.now();
    lid ??= LanguageIdentifier.load({ onProgress: (loaded, total) => post({ type: "progress", loaded, total }) });
    try {
      await lid;
      post({ type: "ready", loadMs: performance.now() - t0 });
    } catch (err) {
      lid = undefined;
      post({ type: "error", message: String(err instanceof Error ? err.message : err) });
    }
    return;
  }

  try {
    const t0 = performance.now();
    const { samples, sampleRate, candidates } = data;
    const result = await (await lid!).identify(samples, { sampleRate, candidates, topK: 5 });
    post({ type: "result", id: data.id, result, ms: performance.now() - t0 });
  } catch (err) {
    post({ type: "error", id: data.id, message: String(err instanceof Error ? err.message : err) });
  }
};
