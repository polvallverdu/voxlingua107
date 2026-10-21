import * as ort from "onnxruntime-node";
import { LanguageIdentifierBase } from "./core.ts";
import { resolveModel, type ModelOptions } from "./model.ts";

export { SAMPLE_RATE, decodeWav, int16ToFloat32, resample, type DecodedAudio } from "./audio.ts";
export type { AudioInput, AudioOptions, IdentifyOptions, IdentifyResult, ModelOutput, Prediction } from "./core.ts";
export { LANGUAGES, isLanguageCode, type Language, type LanguageCode } from "./languages.ts";
export { DEFAULT_MODEL, defaultCacheDir, resolveModel, type ModelOptions } from "./model.ts";

export interface LoadOptions extends ModelOptions {
  /** Passed to `ort.InferenceSession.create` (threads, execution providers, ...). */
  sessionOptions?: ort.InferenceSession.SessionOptions;
  /** Run one dummy inference before returning, so the first `identify` is fast. Default false. */
  warmup?: boolean;
}

/** Spoken language identification for 107 languages (SpeechBrain VoxLingua107 ECAPA-TDNN). */
export class LanguageIdentifier extends LanguageIdentifierBase {
  /** Load the model, downloading it into the cache on first use (~87 MB). */
  static async load(options: LoadOptions = {}): Promise<LanguageIdentifier> {
    const path = await resolveModel(options);
    const lid = new LanguageIdentifier(await ort.InferenceSession.create(path, options.sessionOptions), ort.Tensor);
    if (options.warmup) await lid.warmup();
    return lid;
  }
}
