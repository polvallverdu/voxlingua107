// Builds the demo + docs site (site/) into site-dist/ for GitHub Pages.
//   bun run site         build
//   bun run site:dev     build and serve on http://localhost:3000
import { copyFile, readdir, rm } from "node:fs/promises";
import { basename, join } from "node:path";

const root = join(import.meta.dir, "..");
const outdir = join(root, "site-dist");

await rm(outdir, { recursive: true, force: true });
const common = { outdir, target: "browser", minify: true } as const;

// The worker gets a content-hashed name so browsers can't pair a new page with a stale cached
// worker (GitHub Pages caches files for 10 minutes). The page learns the name via `define`.
const workerBuild = await Bun.build({
  ...common,
  entrypoints: [join(root, "site", "worker.ts")],
  naming: "[name]-[hash].[ext]",
});
if (!workerBuild.success) throw new AggregateError(workerBuild.logs, "Worker build failed");
const workerFile = basename(workerBuild.outputs.find((o) => o.kind === "entry-point")!.path);

const build = await Bun.build({
  ...common,
  entrypoints: [join(root, "site", "index.html")],
  // Relative URLs: the site is served from a subpath (/voxlingua107/) on GitHub Pages.
  publicPath: "./",
  naming: { entry: "[name].[ext]", chunk: "[name]-[hash].[ext]", asset: "[name]-[hash].[ext]" },
  define: { "process.env.WORKER_FILE": JSON.stringify(workerFile) },
});
if (!build.success) throw new AggregateError(build.logs, "Site build failed");

// onnxruntime-web loads its .wasm from next to the script that imports it (the worker). The default
// bundle uses the WebGPU-capable "jsep" build.
const ortDist = join(root, "node_modules", "onnxruntime-web", "dist");
const wasm = "ort-wasm-simd-threaded.jsep.wasm";
await copyFile(join(ortDist, wasm), join(outdir, wasm));
await copyFile(join(root, "test", "fixtures", "udhr_th.wav"), join(outdir, "udhr_th.wav"));

console.log(`Built ${outdir}:`, (await readdir(outdir)).join(", "));

if (process.argv.includes("--serve")) {
  const server = Bun.serve({
    port: 3000,
    async fetch(req) {
      const path = new URL(req.url).pathname;
      const file = Bun.file(join(outdir, path === "/" ? "index.html" : path));
      return (await file.exists()) ? new Response(file) : new Response("Not found", { status: 404 });
    },
  });
  console.log(`Serving on ${server.url}`);
}
