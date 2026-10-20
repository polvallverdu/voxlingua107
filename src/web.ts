// Browser entry point (`voxlingua107/web`): onnxruntime-web (WebAssembly), Fetch, Cache Storage
// and Web Crypto. Must not import node:* modules.
import * as ort from "onnxruntime-web";
import { LanguageIdentifierBase } from "./core.ts";
import { DEFAULT_MODEL } from "./source.ts";

export { SAMPLE_RATE, decodeWav, int16ToFloat32, resample, type DecodedAudio } from "./audio.ts";
export type { AudioInput, AudioOptions, IdentifyOptions, IdentifyResult, ModelOutput, Prediction } from "./core.ts";
export { LANGUAGES, isLanguageCode, type Language, type LanguageCode } from "./languages.ts";
export { DEFAULT_MODEL } from "./source.ts";

export interface FetchModelOptions {
  /** Download URLs, tried in order. Defaults to the Hugging Face mirror, then the original export. */
  urls?: readonly string[];
  /** Expected SHA-256 of the model file. Pass `null` to skip verification (custom models). */
  sha256?: string | null;
  /**
   * Keep the downloaded model in Cache Storage so later page loads skip the ~87 MB download.
   * `true` (default) uses the cache named "voxlingua107"; a string picks another name.
   */
  cache?: boolean | string;
  /** Download progress callback. `total` is 0 when the server doesn't send a length. */
  onProgress?: (loaded: number, total: number) => void;
}

export interface WebLoadOptions extends FetchModelOptions {
  /** Model bytes you already have (bundled asset, IndexedDB, ...). When set, nothing is fetched. */
  model?: ArrayBuffer | Uint8Array;
  /**
   * Where onnxruntime-web loads its `.wasm` files from. Set this if your bundler doesn't serve them,
   * e.g. `"https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/"`. Sets `ort.env.wasm.wasmPaths`.
   */
  wasmPaths?: ort.Env.WasmPrefixOrFilePaths;
  /**
   * WebAssembly threads. Threads need a cross-origin isolated page (COOP/COEP headers) and a
   * bundler setup where onnxruntime-web can start its worker. Sets `ort.env.wasm.numThreads`.
   * Default 1: works everywhere.
   */
  numThreads?: number;
  /** Passed to `ort.InferenceSession.create` (e.g. `{ executionProviders: ["webgpu"] }`). */
  sessionOptions?: ort.InferenceSession.SessionOptions;
}

/** Spoken language identification for 107 languages, in the browser (WebAssembly). */
export class LanguageIdentifier extends LanguageIdentifierBase {
  /** Load the model, fetching it on first use (~87 MB) and caching it in Cache Storage. */
  static async load(options: WebLoadOptions = {}): Promise<LanguageIdentifier> {
    if (options.wasmPaths) ort.env.wasm.wasmPaths = options.wasmPaths;
    ort.env.wasm.numThreads = options.numThreads ?? 1;
    const model = options.model
      ? options.model instanceof Uint8Array
        ? options.model
        : new Uint8Array(options.model)
      : await fetchModel(options);
    return new LanguageIdentifier(await ort.InferenceSession.create(model, options.sessionOptions), ort.Tensor);
  }
}

const inFlight = new Map<string, Promise<Uint8Array>>();

/** Fetch and verify the model bytes, using Cache Storage when available. */
export function fetchModel(options: FetchModelOptions = {}): Promise<Uint8Array> {
  const sha256 = options.sha256 === undefined ? DEFAULT_MODEL.sha256 : options.sha256;
  const urls = options.urls ?? DEFAULT_MODEL.urls;
  const cacheName = options.cache === false ? null : typeof options.cache === "string" ? options.cache : "voxlingua107";
  // Content-addressed key: the same file is reused whichever URL it came from.
  const key = `https://voxlingua107.invalid/${sha256 ?? `unverified/${encodeURIComponent(urls.join(","))}`}`;

  let pending = inFlight.get(key);
  if (!pending) {
    pending = load(urls, sha256, cacheName, key, options.onProgress).finally(() => inFlight.delete(key));
    inFlight.set(key, pending);
  }
  return pending;
}

async function load(
  urls: readonly string[],
  sha256: string | null,
  cacheName: string | null,
  key: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<Uint8Array> {
  const cache = cacheName && typeof caches !== "undefined" ? await caches.open(cacheName).catch(() => null) : null;
  const cached = await cache?.match(key);
  if (cached) return new Uint8Array(await cached.arrayBuffer());

  if (sha256 && !globalThis.crypto?.subtle) {
    throw new Error(
      "Can't verify the model: Web Crypto is only available on https:// and localhost. " +
        "Serve the page securely, or pass `sha256: null` to skip verification.",
    );
  }

  const errors: string[] = [];
  for (const url of urls) {
    try {
      const bytes = await download(url, onProgress);
      if (sha256) {
        const actual = toHex(await crypto.subtle.digest("SHA-256", bytes));
        if (actual !== sha256) throw new Error(`SHA-256 mismatch (got ${actual})`);
      }
      // Best effort: storage can be full or disabled (private browsing).
      await cache?.put(key, new Response(bytes)).catch(() => {});
      return bytes;
    } catch (err) {
      errors.push(`${url}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  throw new Error(`Failed to download the VoxLingua107 model:\n  ${errors.join("\n  ")}`);
}

async function download(
  url: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<Uint8Array<ArrayBuffer>> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;

  const chunks: Uint8Array[] = [];
  let loaded = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function toHex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), (b) => b.toString(16).padStart(2, "0")).join("");
}
