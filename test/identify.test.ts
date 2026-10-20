import { afterAll, beforeAll, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { LanguageIdentifier, decodeWav, isLanguageCode, resample } from "../src/index.ts";

const { samples } = decodeWav(await readFile(join(import.meta.dir, "fixtures", "th_full.wav")));

let lid: LanguageIdentifier;
beforeAll(async () => {
  lid = await LanguageIdentifier.load();
}, 300_000);
afterAll(() => lid?.dispose());

test("returns the best language and a ranked top-k", async () => {
  const result = await lid.identify(samples, { topK: 3 });
  expect(result).toMatchObject({ code: "th", name: "Thai" });
  expect(result.top).toHaveLength(3);
  expect(result.top[0]).toEqual({ code: result.code, name: result.name, probability: result.probability });
  expect(result.top[0]!.probability).toBeGreaterThan(result.top[1]!.probability);
});

test("candidates restrict and renormalize", async () => {
  const result = await lid.identify(samples, { candidates: ["en", "es", "fr"], topK: 10 });
  expect(result.top.map((p) => p.code).sort()).toEqual(["en", "es", "fr"]);
  expect(result.top.reduce((a, p) => a + p.probability, 0)).toBeCloseTo(1, 5);
  // @ts-expect-error -- "xx" isn't a LanguageCode; JS callers still get a runtime error.
  await expect(lid.identify(samples, { candidates: ["xx"] })).rejects.toThrow('Unknown language code "xx"');
});

test("accepts 16-bit PCM", async () => {
  const pcm = Int16Array.from(samples, (s) => Math.round(s * 32767));
  expect((await lid.identify(pcm)).code).toBe("th");
});

test("resamples other sample rates", async () => {
  const at48k = resample(samples, 16000, 48000);
  const native = await lid.identify(samples);
  const resampled = await lid.identify(at48k, { sampleRate: 48000 });
  expect(resampled.code).toBe("th");
  expect(Math.abs(resampled.probability - native.probability)).toBeLessThan(0.01);
});

test("embedding has 256 dimensions", async () => {
  expect(await lid.embed(samples)).toHaveLength(256);
});

test("rejects audio that is too short", async () => {
  await expect(lid.identify(new Float32Array(100))).rejects.toThrow("Audio too short");
});

test("isLanguageCode narrows strings", () => {
  expect(isLanguageCode("ca")).toBe(true);
  expect(isLanguageCode("he")).toBe(true);
  expect(isLanguageCode("iw")).toBe(false); // exposed as "he"
  expect(isLanguageCode("xx")).toBe(false);
});
