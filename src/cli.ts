#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { LanguageIdentifier, SAMPLE_RATE, decodeWav, resample, resolveModel, type ModelOptions } from "./index.ts";

const HELP = `Usage:
  voxlingua107 download [--cache-dir <dir>]
      Download the model into the cache and print its path.
  voxlingua107 identify <audio-file> [--top <n>] [--cache-dir <dir>] [--model <path>]
      Identify the spoken language. WAV is read directly; other formats need ffmpeg on PATH.
`;

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    "cache-dir": { type: "string" },
    model: { type: "string" },
    top: { type: "string", default: "5" },
    help: { type: "boolean", short: "h" },
  },
});

const [command, file] = positionals;
const modelOptions: ModelOptions = {
  cacheDir: values["cache-dir"],
  modelPath: values.model,
  onProgress: process.stderr.isTTY ? progress : undefined,
};

if (values.help || !command) {
  process.stdout.write(HELP);
} else if (command === "download") {
  console.log(await resolveModel(modelOptions));
} else if (command === "identify" && file) {
  const samples = await loadAudio(file);
  const lid = await LanguageIdentifier.load(modelOptions);
  const result = await lid.identify(samples, { topK: Number(values.top) });
  for (const p of result.top) {
    console.log(`${p.code.padEnd(4)} ${p.name.padEnd(18)} ${(p.probability * 100).toFixed(1)}%`);
  }
  await lid.dispose();
} else {
  process.stderr.write(HELP);
  process.exit(1);
}

async function loadAudio(path: string): Promise<Float32Array> {
  if (path.toLowerCase().endsWith(".wav")) {
    const { samples, sampleRate } = decodeWav(await readFile(path));
    return resample(samples, sampleRate, SAMPLE_RATE);
  }
  const ff = spawnSync("ffmpeg", ["-v", "error", "-i", path, "-ac", "1", "-ar", String(SAMPLE_RATE), "-f", "f32le", "-"], {
    maxBuffer: 1 << 30,
  });
  if (ff.error || ff.status !== 0) {
    throw new Error(`Couldn't decode ${path} with ffmpeg: ${ff.error?.message ?? ff.stderr.toString()}`);
  }
  const buf = ff.stdout;
  return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

function progress(loaded: number, total: number) {
  const mb = (n: number) => (n / 1e6).toFixed(1);
  process.stderr.write(`\rDownloading model ${mb(loaded)}${total ? ` / ${mb(total)}` : ""} MB`);
  if (total && loaded >= total) process.stderr.write("\n");
}
