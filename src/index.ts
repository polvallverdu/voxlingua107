import * as ort from "onnxruntime-node";
import { SAMPLE_RATE, int16ToFloat32, resample } from "./audio.ts";
import { LANGUAGES } from "./languages.ts";
import { resolveModel, type ModelOptions } from "./model.ts";

export { SAMPLE_RATE, decodeWav, int16ToFloat32, resample, type DecodedAudio } from "./audio.ts";
export { LANGUAGES, type Language } from "./languages.ts";
export { DEFAULT_MODEL, defaultCacheDir, resolveModel, type ModelOptions } from "./model.ts";

export interface LoadOptions extends ModelOptions {
  /** Passed to `ort.InferenceSession.create` (threads, execution providers, ...). */
  sessionOptions?: ort.InferenceSession.SessionOptions;
}

/** Audio as float samples in [-1, 1] or 16-bit PCM, mono. */
export type AudioInput = Float32Array | Int16Array;

export interface AudioOptions {
  /** Sample rate of the input. Anything other than 16000 is resampled. Default 16000. */
  sampleRate?: number;
}

export interface IdentifyOptions extends AudioOptions {
  /** Number of ranked predictions to return in `top`. Default 5. */
  topK?: number;
  /**
   * Only consider these language codes (e.g. the languages your app supports).
   * Probabilities are renormalized over this set.
   */
  candidates?: readonly string[];
}

export interface Prediction {
  /** ISO 639 code, e.g. "en". */
  code: string;
  /** English name, e.g. "English". */
  name: string;
  /** Probability in [0, 1]. */
  probability: number;
}

export interface IdentifyResult extends Prediction {
  /** The `topK` most likely languages, best first (the first one is the result itself). */
  top: Prediction[];
}

/** Spoken language identification for 107 languages (SpeechBrain VoxLingua107 ECAPA-TDNN). */
export class LanguageIdentifier {
  private constructor(private readonly session: ort.InferenceSession) {}

  /** Load the model, downloading it into the cache on first use (~87 MB). */
  static async load(options: LoadOptions = {}): Promise<LanguageIdentifier> {
    const path = await resolveModel(options);
    return new LanguageIdentifier(await ort.InferenceSession.create(path, options.sessionOptions));
  }

  /** Identify the spoken language. A few seconds of speech is enough; under ~3 s gets less reliable. */
  async identify(audio: AudioInput, options: IdentifyOptions = {}): Promise<IdentifyResult> {
    const indices = candidateIndices(options.candidates);
    const { logits } = await this.run(audio, options);

    // Softmax over the allowed classes (log-sum-exp for stability).
    let max = -Infinity;
    for (const i of indices) max = Math.max(max, logits[i]!);
    let sum = 0;
    for (const i of indices) sum += Math.exp(logits[i]! - max);

    const ranked = indices
      .map((i) => ({ ...LANGUAGES[i]!, probability: Math.exp(logits[i]! - max) / sum }))
      .sort((a, b) => b.probability - a.probability);

    const top = ranked.slice(0, Math.max(1, options.topK ?? 5));
    return { ...top[0]!, top };
  }

  /** The 256-dim language embedding (useful for clustering or similarity). */
  async embed(audio: AudioInput, options: AudioOptions = {}): Promise<Float32Array> {
    return (await this.run(audio, options)).embedding;
  }

  /** Raw model outputs: per-class log-probabilities (in `LANGUAGES` order) and the embedding. */
  async run(
    audio: AudioInput,
    options: AudioOptions = {},
  ): Promise<{ logits: Float32Array; embedding: Float32Array }> {
    let samples = audio instanceof Int16Array ? int16ToFloat32(audio) : audio;
    samples = resample(samples, options.sampleRate ?? SAMPLE_RATE, SAMPLE_RATE);
    if (samples.length < SAMPLE_RATE / 10) {
      throw new Error(`Audio too short (${samples.length} samples at 16 kHz); need at least 0.1 s`);
    }

    const input = new ort.Tensor("float32", samples, [1, samples.length]);
    const out = await this.session.run({ audio: input });
    return {
      logits: out.logits!.data as Float32Array,
      embedding: out.embedding!.data as Float32Array,
    };
  }

  /** Release the ONNX Runtime session. */
  async dispose(): Promise<void> {
    await this.session.release();
  }
}

function candidateIndices(candidates?: readonly string[]): number[] {
  if (!candidates) return LANGUAGES.map((_, i) => i);
  const indices = candidates.map((code) => {
    const i = LANGUAGES.findIndex((l) => l.code === code);
    if (i < 0) throw new Error(`Unknown language code "${code}"`);
    return i;
  });
  if (!indices.length) throw new Error("`candidates` must not be empty");
  return [...new Set(indices)];
}
