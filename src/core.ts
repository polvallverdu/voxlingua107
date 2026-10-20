// Runtime-independent part of the identifier, shared by the Node (onnxruntime-node) and
// web (onnxruntime-web) entry points. Must not import node:* modules.
import { SAMPLE_RATE, int16ToFloat32, resample } from "./audio.ts";
import { LANGUAGES, type LanguageCode } from "./languages.ts";

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
  candidates?: readonly LanguageCode[];
}

export interface Prediction {
  /** ISO 639 code, e.g. "en". */
  code: LanguageCode;
  /** English name, e.g. "English". */
  name: string;
  /** Probability in [0, 1]. */
  probability: number;
}

export interface IdentifyResult extends Prediction {
  /** The `topK` most likely languages, best first (the first one is the result itself). */
  top: Prediction[];
}

export interface ModelOutput {
  /** Per-class log-probabilities, in `LANGUAGES` order. */
  logits: Float32Array;
  /** 256-dim language embedding. */
  embedding: Float32Array;
}

/** Minimal slice of the ONNX Runtime API that both onnxruntime-node and onnxruntime-web provide. */
export interface OrtSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: unknown } | undefined>>;
  release(): Promise<void>;
}
export type OrtTensorConstructor = new (type: "float32", data: Float32Array, dims: readonly number[]) => unknown;

/** Spoken language identification for 107 languages (SpeechBrain VoxLingua107 ECAPA-TDNN). */
export class LanguageIdentifierBase {
  protected constructor(
    private readonly session: OrtSession,
    private readonly Tensor: OrtTensorConstructor,
  ) {}

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
  async run(audio: AudioInput, options: AudioOptions = {}): Promise<ModelOutput> {
    let samples = audio instanceof Int16Array ? int16ToFloat32(audio) : audio;
    samples = resample(samples, options.sampleRate ?? SAMPLE_RATE, SAMPLE_RATE);
    if (samples.length < SAMPLE_RATE / 10) {
      throw new Error(`Audio too short (${samples.length} samples at 16 kHz); need at least 0.1 s`);
    }

    const input = new this.Tensor("float32", samples, [1, samples.length]);
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

function candidateIndices(candidates?: readonly LanguageCode[]): number[] {
  if (!candidates) return LANGUAGES.map((_, i) => i);
  const indices = candidates.map((code) => {
    const i = LANGUAGES.findIndex((l) => l.code === code);
    if (i < 0) throw new Error(`Unknown language code "${code}"`);
    return i;
  });
  if (!indices.length) throw new Error("`candidates` must not be empty");
  return [...new Set(indices)];
}
