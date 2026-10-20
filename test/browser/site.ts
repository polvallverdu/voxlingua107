// End-to-end check of the demo site (site-dist/) in headless Chromium: the sample button and the
// live microphone mode, with Chromium's fake mic playing the Thai test clip.
// Run with `bun run test:site` (builds the site first).
import { join } from "node:path";
import { chromium } from "playwright";
import { resolveModel } from "../../src/index.ts";

const root = join(import.meta.dir, "..", "..");
const outdir = join(root, "site-dist");
const modelPath = await resolveModel();

const requested: string[] = [];
const server = Bun.serve({
  port: 0,
  async fetch(req) {
    const path = new URL(req.url).pathname;
    requested.push(path);
    if (process.env.DEBUG && /\.(wasm|mjs)$/.test(path)) console.log("[server]", path);
    if (path === "/model.onnx") return new Response(Bun.file(modelPath), { headers: { "access-control-allow-origin": "*" } });
    const file = Bun.file(join(outdir, path === "/" ? "index.html" : path));
    return (await file.exists()) ? new Response(file) : new Response("Not found", { status: 404 });
  },
});

const args = [
  "--use-fake-ui-for-media-stream",
  "--use-fake-device-for-media-stream",
  `--use-file-for-fake-audio-capture=${join(root, "test", "fixtures", "udhr_th.wav")}`,
  "--autoplay-policy=no-user-gesture-required",
];
const browser = await chromium.launch({ args }).catch(() => chromium.launch({ args, channel: "chrome" }));
try {
  const page = await browser.newPage();
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message));
  page.on("crash", () => console.error("[page crashed]"));
  if (process.env.DEBUG) page.on("worker", (w) => w.on("console", (m) => console.log("[worker]", m.text())));
  if (process.env.DEBUG) page.on("console", (msg) => console.log("[browser]", msg.type(), msg.text()));
  page.on("console", (msg) => msg.type() === "error" && errors.push(msg.text()));
  // Redirect the Hugging Face download to the local server instead of fetching 87 MB.
  // (Fulfilling the route with the body itself kills the browser connection at this size.)
  let modelFetches = 0;
  await page.context().route(/huggingface\.co\/.*\.onnx$/, async (route) => {
    modelFetches++;
    await route.fulfill({
      status: 302,
      headers: { location: `http://localhost:${server.port}/model.onnx`, "access-control-allow-origin": "*" },
    });
  });

  await page.goto(`http://localhost:${server.port}/`);

  await page.click("#sample");
  await page.waitForFunction(() => document.getElementById("language")?.textContent === "Thai", null, {
    timeout: 120_000,
  });
  console.log(`sample button: ${await page.textContent("#confidence")}`);

  await page.click("#mic");
  await page.waitForFunction(() => document.getElementById("status")?.textContent?.startsWith("Listening"), null, {
    timeout: 30_000,
  });
  await page.evaluate(() => (document.getElementById("language")!.textContent = "")); // reset
  await page.waitForFunction(() => document.getElementById("language")?.textContent === "Thai", null, {
    timeout: 60_000,
  });
  console.log(`live microphone: ${await page.textContent("#confidence")}`);
  await page.click("#mic");

  if (modelFetches !== 1) throw new Error(`Model fetched ${modelFetches} times, expected 1`);
  if (errors.length) throw new Error(`Browser errors:\n${errors.join("\n")}`);
  const wasm = requested.filter((p) => p.endsWith(".wasm") || p.endsWith(".mjs"));
  console.log(`runtime files fetched: ${wasm.join(", ")}`);
  console.log("site test passed");
} finally {
  await browser.close();
  server.stop(true);
}
