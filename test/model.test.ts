// Download/cache logic against a local HTTP server; doesn't touch the real model.
import { afterAll, beforeEach, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { defaultCacheDir, resolveModel } from "../src/model.ts";

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

let cacheDir: string;
const tempDirs: string[] = [];
beforeEach(async () => {
  cacheDir = await mkdtemp(join(tmpdir(), "voxlingua107-test-"));
  tempDirs.push(cacheDir);
  for (const k of Object.keys(hits)) delete hits[k];
});
afterAll(() => server.stop(true));

test("downloads, verifies and caches", async () => {
  const path = await resolveModel({ cacheDir, sha256, urls: [url("/good")] });
  expect(path.startsWith(cacheDir)).toBe(true);
  expect(await readFile(path)).toEqual(Buffer.from(payload));

  await resolveModel({ cacheDir, sha256, urls: [url("/good")] });
  expect(hits["/good"]).toBe(1);
  expect(await readdir(cacheDir)).toHaveLength(1); // no temp files left behind
});

test("falls back to the next URL on HTTP errors and hash mismatches", async () => {
  const path = await resolveModel({ cacheDir, sha256, urls: [url("/missing"), url("/corrupt"), url("/good")] });
  expect(await readFile(path)).toEqual(Buffer.from(payload));
  expect(hits).toEqual({ "/missing": 1, "/corrupt": 1, "/good": 1 });
});

test("rejects a file whose hash doesn't match", async () => {
  await expect(resolveModel({ cacheDir, sha256, urls: [url("/corrupt")] })).rejects.toThrow("SHA-256 mismatch");
  expect(await readdir(cacheDir)).toHaveLength(0);
});

test("sha256: null skips verification", async () => {
  const path = await resolveModel({ cacheDir, sha256: null, urls: [url("/corrupt")] });
  expect(await readFile(path, "utf8")).toBe("tampered");
});

test("concurrent loads share one download", async () => {
  const opts = { cacheDir, sha256, urls: [url("/good")] };
  const paths = await Promise.all([resolveModel(opts), resolveModel(opts), resolveModel(opts)]);
  expect(new Set(paths).size).toBe(1);
  expect(hits["/good"]).toBe(1);
});

test("download: false fails when not cached", async () => {
  await expect(resolveModel({ cacheDir, sha256, download: false })).rejects.toThrow("downloads are disabled");
});

test("modelPath is used as-is", async () => {
  const file = join(cacheDir, "custom.onnx");
  await writeFile(file, payload);
  expect(await resolveModel({ modelPath: file, urls: [url("/good")] })).toBe(file);
  expect(hits["/good"]).toBeUndefined();
  await expect(resolveModel({ modelPath: join(cacheDir, "missing.onnx") })).rejects.toThrow("Model not found");
});

test("cache dir respects VOXLINGUA107_CACHE_DIR and XDG_CACHE_HOME", () => {
  const saved = { ...process.env };
  try {
    process.env.VOXLINGUA107_CACHE_DIR = "/models/lid";
    expect(defaultCacheDir()).toBe("/models/lid");
    delete process.env.VOXLINGUA107_CACHE_DIR;
    process.env.XDG_CACHE_HOME = "/xdg";
    expect(defaultCacheDir()).toBe(join("/xdg", "voxlingua107"));
  } finally {
    process.env = saved;
  }
});

afterAll(async () => {
  await Promise.all(tempDirs.map((d) => rm(d, { recursive: true, force: true })));
});
