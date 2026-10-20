import { describe, expect, test } from "bun:test";
import { decodeWav, int16ToFloat32, resample } from "../src/audio.ts";
import { encodeWav, sine } from "./helpers.ts";

describe("decodeWav", () => {
  const signal = Float32Array.from([0, 0.5, -0.5, 0.25, -1]);

  for (const format of ["pcm16", "pcm24", "float32"] as const) {
    test(format, () => {
      const { samples, sampleRate } = decodeWav(encodeWav(signal, { sampleRate: 22050, format }));
      expect(sampleRate).toBe(22050);
      expect(samples.length).toBe(signal.length);
      samples.forEach((s, i) => expect(s).toBeCloseTo(signal[i]!, 3));
    });
  }

  test("averages channels to mono", () => {
    const stereo = Float32Array.from([1, 0, 0.5, 0.5, -1, 0]); // L/R interleaved
    const { samples } = decodeWav(encodeWav(stereo, { channels: 2, format: "float32" }));
    expect(Array.from(samples)).toEqual([0.5, 0.5, -0.5]);
  });

  test("accepts Node Buffers and other views", () => {
    const file = encodeWav(signal);
    const padded = new Uint8Array(file.length + 8);
    padded.set(file, 8);
    expect(decodeWav(padded.subarray(8)).samples.length).toBe(signal.length);
    expect(decodeWav(Buffer.from(file)).samples.length).toBe(signal.length);
  });

  test("rejects non-WAV data", () => {
    expect(() => decodeWav(new Uint8Array(64))).toThrow("Not a RIFF/WAVE file");
  });
});

test("int16ToFloat32", () => {
  expect(Array.from(int16ToFloat32(Int16Array.from([0, 16384, -32768])))).toEqual([0, 0.5, -1]);
});

describe("resample", () => {
  // Estimate a pure tone's frequency from its zero crossings.
  const frequency = (x: Float32Array, rate: number) => {
    let crossings = 0;
    for (let i = 1; i < x.length; i++) if (x[i - 1]! < 0 !== x[i]! < 0) crossings++;
    return (crossings / 2) * (rate / x.length);
  };

  test("returns the input when rates match", () => {
    const x = sine(440, 0.1, 16000);
    expect(resample(x, 16000, 16000)).toBe(x);
  });

  for (const from of [8000, 22050, 44100, 48000]) {
    test(`${from} Hz -> 16 kHz keeps length and pitch`, () => {
      const out = resample(sine(440, 1, from), from, 16000);
      expect(out.length).toBe(16000);
      expect(Math.abs(frequency(out, 16000) - 440)).toBeLessThan(2);
    });
  }

  test("removes content above the new Nyquist when downsampling", () => {
    const out = resample(sine(12000, 1, 48000), 48000, 16000); // 12 kHz > 8 kHz Nyquist
    const rms = Math.sqrt(out.slice(1000, -1000).reduce((a, s) => a + s * s, 0) / (out.length - 2000));
    expect(rms).toBeLessThan(0.01);
  });
});
