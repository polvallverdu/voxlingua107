// End-to-end check of `voxlingua107/web` in headless Chromium: bundles test/browser/page.ts with
// Bun's bundler, serves it with the model and a test clip, and checks the result twice (the
// second page load must come from Cache Storage). Run with `bun run test:browser`.
import { readdir } from "node:fs/promises";
import { join } from "node:path";
import { chromium } from "playwright";
import { resolveModel } from "../../src/index.ts";

const outdir = join(import.meta.dir, "..", "..", ".cache", "browser-test");
const build = await Bun.build({ entrypoints: [join(import.meta.dir, "page.ts")], outdir, target: "browser" });
if (!build.success) throw new AggregateError(build.logs, "Bundling failed");

// onnxruntime-web fetches its .wasm next to the bundle (new URL(..., import.meta.url)).
// Bun's bundler doesn't copy it, so serve it from the package like an app's static assets would.
const ortDist = join(import.meta.dir, "..", "..", "node_modules", "onnxruntime-web", "dist");
const wasmFiles = new Set((await readdir(ortDist)).filter((f) => f.endsWith(".wasm")));

const modelPath = await resolveModel();
const modelRequests: string[] = [];
const server = Bun.serve({
  port: 0,
  async fetch(req) {
    const path = new URL(req.url).pathname;
    const file = path.slice(1);
    if (path === "/") return new Response(`<script type="module" src="/page.js"></script>`, { headers: { "content-type": "text/html" } });
    if (path === "/page.js") return new Response(Bun.file(join(outdir, "page.js")));
    if (path === "/model.onnx") {
      modelRequests.push(path);
      return new Response(Bun.file(modelPath));
    }
    if (path === "/udhr_th.wav") return new Response(Bun.file(join(import.meta.dir, "..", "fixtures", "udhr_th.wav")));
    if (wasmFiles.has(file)) return new Response(Bun.file(join(ortDist, file)));
    return new Response("not found", { status: 404 });
  },
});

const browser = await chromium.launch().catch(() => chromium.launch({ channel: "chrome" }));
try {
  const page = await browser.newPage();
  page.on("console", (msg) => msg.type() === "error" && console.error("[browser]", msg.text()));

  for (const attempt of ["first load (download)", "second load (Cache Storage)"]) {
    await page.goto(`http://localhost:${server.port}/`);
    await page.waitForFunction(() => window.result !== undefined, null, { timeout: 120_000 });
    const out = (await page.evaluate(() => window.result)) as {
      ok: boolean;
      error?: string;
      result?: { code: string; probability: number };
      loadMs?: number;
      identifyMs?: number;
    };
    if (!out.ok) throw new Error(`${attempt}: ${out.error}`);
    const { code, probability } = out.result!;
    console.log(
      `${attempt}: ${code} ${(probability * 100).toFixed(1)}% | load ${out.loadMs!.toFixed(0)} ms, identify 13 s clip ${out.identifyMs!.toFixed(0)} ms`,
    );
    if (code !== "th" || probability < 0.95) throw new Error(`${attempt}: expected Thai, got ${code} (${probability})`);
  }

  if (modelRequests.length !== 1) throw new Error(`Model fetched ${modelRequests.length} times; expected 1 (cached)`);
  console.log("browser test passed");
} finally {
  await browser.close();
  server.stop(true);
}
