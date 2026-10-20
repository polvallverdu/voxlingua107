// The browser entry point (onnxruntime-web / WebAssembly), run under Bun.
import { afterAll, beforeAll, beforeEach, describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { resolveModel } from "../src/index.ts";
import { LanguageIdentifier, decodeWav, fetchModel } from "../src/web.ts";

const fixtures = join(import.meta.dir, "fixtures");
const reference = JSON.parse(await readFile(join(fixtures, "reference.json"), "utf8")) as {
  clips: Record<string, { label: string; logprobs: number[] }>;
};

describe("matches PyTorch (WebAssembly)", () => {
  let lid: LanguageIdentifier;
  beforeAll(async () => {
    // Reuse the Node cache so the test doesn't download the model twice.
    lid = await LanguageIdentifier.load({ model: await readFile(await resolveModel()) });
  }, 300_000);
  afterAll(() => lid?.dispose());

  for (const name of ["th_full.wav", "th_1s.wav", "noise_4s.wav"]) {
    test(name, async () => {
      const ref = reference.clips[name]!;
      const { samples } = decodeWav(await readFile(join(fixtures, name)));
      const { logits } = await lid.run(samples);

      let maxProbDiff = 0;
      for (let i = 0; i < logits.length; i++) {
        maxProbDiff = Math.max(maxProbDiff, Math.abs(Math.exp(logits[i]!) - Math.exp(ref.logprobs[i]!)));
      }
      expect(maxProbDiff).toBeLessThan(2e-5);

      const result = await lid.identify(samples);
      expect(`${result.code}: ${result.name}`).toBe(ref.label);
    }, 60_000);
  }
});

describe("fetchModel", () => {
  const payload = new TextEncoder().encode("not really an onnx model");
  const sha256 = createHash("sha256").update(payload).digest("hex");

  const hits: Record<string, number> = {};
  const server = Bun.serve({
    port: 0,
    fetch(req) {
      const path = new URL(req.url).pathname;
      hits[path] = (hits[path] ?? 0) + 1;
      if (path === "/good") return new Response(payload);
      if (path === "/corrupt") return new Response("tampered");
      return new Response("nope", { status: 404 });
    },
  });
  const url = (path: string) => `http://localhost:${server.port}${path}`;

  // Minimal in-memory Cache Storage (Bun doesn't have one).
  const stores = new Map<string, Map<string, Response>>();
  const fakeCaches = {
    async open(name: string) {
      const store = stores.get(name) ?? new Map<string, Response>();
      stores.set(name, store);
      return {
        match: async (key: string) => store.get(key)?.clone(),
        put: async (key: string, res: Response) => void store.set(key, res),
      };
    },
  };

  beforeEach(() => {
    for (const k of Object.keys(hits)) delete hits[k];
    stores.clear();
    (globalThis as { caches?: unknown }).caches = fakeCaches;
  });
  afterAll(() => {
    delete (globalThis as { caches?: unknown }).caches;
    server.stop(true);
  });

  test("verifies, caches, and serves later loads from the cache", async () => {
    expect(await fetchModel({ sha256, urls: [url("/good")] })).toEqual(payload);
    expect(await fetchModel({ sha256, urls: [url("/good")] })).toEqual(payload);
    expect(hits["/good"]).toBe(1);
    expect(stores.get("voxlingua107")?.size).toBe(1);
  });

  test("falls back to the next URL on HTTP errors and hash mismatches", async () => {
    expect(await fetchModel({ sha256, urls: [url("/missing"), url("/corrupt"), url("/good")] })).toEqual(payload);
    expect(hits).toEqual({ "/missing": 1, "/corrupt": 1, "/good": 1 });
  });

  test("rejects a file whose hash doesn't match and doesn't cache it", async () => {
    await expect(fetchModel({ sha256, urls: [url("/corrupt")] })).rejects.toThrow("SHA-256 mismatch");
    expect(stores.get("voxlingua107")?.size ?? 0).toBe(0);
  });

  test("cache: false skips Cache Storage; a string picks the cache name", async () => {
    await fetchModel({ sha256, urls: [url("/good")], cache: false });
    expect(stores.size).toBe(0);
    await fetchModel({ sha256, urls: [url("/good")], cache: "my-app" });
    expect([...stores.keys()]).toEqual(["my-app"]);
  });

  test("works without Cache Storage", async () => {
    delete (globalThis as { caches?: unknown }).caches;
    expect(await fetchModel({ sha256, urls: [url("/good")] })).toEqual(payload);
  });

  test("reports progress", async () => {
    const calls: number[] = [];
    await fetchModel({ sha256, urls: [url("/good")], cache: false, onProgress: (loaded) => calls.push(loaded) });
    expect(calls.at(-1)).toBe(payload.byteLength);
  });
});
