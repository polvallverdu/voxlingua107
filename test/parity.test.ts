// Checks the ONNX model against the original SpeechBrain (PyTorch) model.
// Reference numbers come from scripts/generate-reference.py.
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { LANGUAGES, LanguageIdentifier, decodeWav, resample } from "../src/index.ts";

const fixtures = join(import.meta.dir, "fixtures");
const reference = JSON.parse(await readFile(join(fixtures, "reference.json"), "utf8")) as {
  labels: string[];
  clips: Record<string, { label: string; logprobs: number[] }>;
};

const wav = async (name: string) => decodeWav(await readFile(join(fixtures, name)));

let lid: LanguageIdentifier;
beforeAll(async () => {
  lid = await LanguageIdentifier.load();
}, 300_000);
afterAll(() => lid?.dispose());

test("label order matches SpeechBrain's label encoder", () => {
  const renamed: Record<string, string> = { iw: "he", jw: "jv" };
  const codes = reference.labels.map((l) => l.split(": ")[0]!);
  expect(LANGUAGES.map((l): string => l.code)).toEqual(codes.map((c) => renamed[c] ?? c));
});

describe("matches PyTorch", () => {
  for (const [name, ref] of Object.entries(reference.clips)) {
    test(name, async () => {
      const { samples, sampleRate } = await wav(name);
      expect(sampleRate).toBe(16000);

      const { logits } = await lid.run(samples);
      let maxLogDiff = 0;
      let maxProbDiff = 0;
      for (let i = 0; i < logits.length; i++) {
        maxLogDiff = Math.max(maxLogDiff, Math.abs(logits[i]! - ref.logprobs[i]!));
        maxProbDiff = Math.max(maxProbDiff, Math.abs(Math.exp(logits[i]!) - Math.exp(ref.logprobs[i]!)));
      }
      // Observed: probabilities within ~6e-6. The largest log-space gaps (~1e-3) are on
      // classes with p < 1e-6, where float32 rounding dominates.
      expect(maxProbDiff).toBeLessThan(2e-5);
      expect(maxLogDiff).toBeLessThan(5e-3);

      const result = await lid.identify(samples);
      expect(`${result.code}: ${result.name}`).toBe(ref.label);
    });
  }
});

test("upstream example clip (44.1 kHz) is Thai after resampling in JS", async () => {
  const { samples, sampleRate } = await wav("udhr_th.wav");
  expect(sampleRate).toBe(44100);

  const result = await lid.identify(resample(samples, sampleRate, 16000));
  const expected = Math.exp(Math.max(...reference.clips["th_full.wav"]!.logprobs));
  expect(result.code).toBe("th");
  // Different resampler than torchaudio, so only approximately equal.
  expect(Math.abs(result.probability - expected)).toBeLessThan(0.01);
});
