// Builds the demo + docs site (site/) into site-dist/ for GitHub Pages.
//   bun run site         build
//   bun run site:dev     build and serve on http://localhost:3000
import { copyFile, readdir, rm } from "node:fs/promises";
import { join } from "node:path";

const root = join(import.meta.dir, "..");
const outdir = join(root, "site-dist");

await rm(outdir, { recursive: true, force: true });
const build = await Bun.build({
  entrypoints: [join(root, "site", "index.html"), join(root, "site", "worker.ts")],
  outdir,
  target: "browser",
  minify: true,
  // Relative URLs: the site is served from a subpath (/voxlingua107/) on GitHub Pages.
  publicPath: "./",
  naming: { entry: "[name].[ext]", chunk: "[name]-[hash].[ext]", asset: "[name]-[hash].[ext]" },
});
if (!build.success) throw new AggregateError(build.logs, "Site build failed");

// onnxruntime-web loads its .wasm from next to the script that imports it (worker.js). The default
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
