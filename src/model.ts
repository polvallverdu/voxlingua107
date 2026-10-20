import { createHash, randomBytes } from "node:crypto";
import { access, mkdir, open, rename, rm } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { DEFAULT_MODEL } from "./source.ts";

export { DEFAULT_MODEL } from "./source.ts";

export interface ModelOptions {
  /** Path to a local `voxlingua107.onnx`. When set, nothing is downloaded or cached. */
  modelPath?: string;
  /**
   * Directory where the model is cached. Defaults to `$VOXLINGUA107_CACHE_DIR`,
   * then `$XDG_CACHE_HOME/voxlingua107`, then `~/.cache/voxlingua107`.
   */
  cacheDir?: string;
  /** Download URLs, tried in order. Defaults to the Hugging Face mirror, then the original export. */
  urls?: readonly string[];
  /** Expected SHA-256 of the model file. Pass `null` to skip verification (custom models). */
  sha256?: string | null;
  /** Set to `false` to fail instead of downloading when the model isn't cached. */
  download?: boolean;
  /** Download progress callback. `total` is 0 when the server doesn't send a length. */
  onProgress?: (loaded: number, total: number) => void;
}

export function defaultCacheDir(): string {
  const env = process.env.VOXLINGUA107_CACHE_DIR;
  if (env) return env;
  const xdg = process.env.XDG_CACHE_HOME;
  return join(xdg || join(homedir(), ".cache"), "voxlingua107");
}

const inFlight = new Map<string, Promise<string>>();

/** Return a local path to the model, downloading and verifying it into the cache if needed. */
export async function resolveModel(options: ModelOptions = {}): Promise<string> {
  if (options.modelPath) {
    if (!(await exists(options.modelPath))) throw new Error(`Model not found at ${options.modelPath}`);
    return options.modelPath;
  }

  const sha256 = options.sha256 === undefined ? DEFAULT_MODEL.sha256 : options.sha256;
  const urls = options.urls ?? DEFAULT_MODEL.urls;
  const cacheDir = options.cacheDir ?? defaultCacheDir();
  // Content-addressed name: a different expected hash never reuses a stale file.
  const file = join(cacheDir, `voxlingua107-${sha256 ? sha256.slice(0, 16) : "unverified"}.onnx`);

  if (await exists(file)) return file;
  if (options.download === false) {
    throw new Error(`Model not cached at ${file} and downloads are disabled`);
  }

  let pending = inFlight.get(file);
  if (!pending) {
    pending = download(urls, file, sha256, options.onProgress).finally(() => inFlight.delete(file));
    inFlight.set(file, pending);
  }
  return pending;
}

async function download(
  urls: readonly string[],
  file: string,
  sha256: string | null,
  onProgress?: (loaded: number, total: number) => void,
): Promise<string> {
  await mkdir(join(file, ".."), { recursive: true });
  const errors: string[] = [];

  for (const url of urls) {
    // Unique temp name + rename, so concurrent processes never see a partial file.
    const tmp = `${file}.${process.pid}.${randomBytes(4).toString("hex")}.tmp`;
    try {
      const res = await fetch(url);
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const total = Number(res.headers.get("content-length")) || 0;
      const hash = createHash("sha256");
      const handle = await open(tmp, "w");
      let loaded = 0;
      try {
        for await (const chunk of res.body) {
          hash.update(chunk);
          await handle.write(chunk);
          loaded += chunk.byteLength;
          onProgress?.(loaded, total);
        }
      } finally {
        await handle.close();
      }
      const actual = hash.digest("hex");
      if (sha256 && actual !== sha256) throw new Error(`SHA-256 mismatch (got ${actual})`);
      await rename(tmp, file);
      return file;
    } catch (err) {
      await rm(tmp, { force: true });
      errors.push(`${url}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  throw new Error(`Failed to download the VoxLingua107 model:\n  ${errors.join("\n  ")}`);
}

async function exists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}
