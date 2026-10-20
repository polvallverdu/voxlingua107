// End-to-end check of the demo site (site-dist/) in headless Chromium: the sample button, the
// language picker, and recording from Chromium's fake mic (which plays the Thai test clip).
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

  // Language picker: excluding Thai, the same clip should be re-analyzed and come out as Lao.
  await page.click("#langs summary");
  await page.check('#lang-list input[value="lo"]');
  await page.check('#lang-list input[value="vi"]');
  await page.waitForFunction(() => document.querySelectorAll("#bars li").length === 2, null, { timeout: 60_000 });
  const bars = await page.$$eval("#bars .name", (els) => els.map((e) => e.textContent));
  if (bars.join() !== "Lao (lo),Vietnamese (vi)") throw new Error(`Unexpected bars with candidates: ${bars}`);
  if ((await page.textContent("#language")) !== "Lao") throw new Error("Expected Lao with Thai excluded");
  console.log(`with candidates [lo, vi]: ${await page.textContent("#confidence")}`);
  await page.click("#lang-all"); // re-analyzes the sample with all languages
  await page.waitForFunction(() => document.querySelectorAll("#bars li").length === 5, null, { timeout: 60_000 });

  // Record from Chromium's fake mic (playing the Thai clip), stop early, and check the result.
  await page.evaluate(() => (document.getElementById("language")!.textContent = "")); // reset
  await page.click("#mic");
  await page.waitForFunction(() => document.getElementById("status")?.textContent?.startsWith("Recording"), null, {
    timeout: 30_000,
  });
  await page.waitForTimeout(5000);
  await page.click("#mic"); // "Stop and analyze"
  await page.waitForFunction(
    () => document.getElementById("status")?.textContent?.startsWith("Result for your recording"),
    null,
    { timeout: 60_000 },
  );
  if ((await page.textContent("#language")) !== "Thai") throw new Error("Recording wasn't identified as Thai");
  if (!(await page.$("#playback audio"))) throw new Error("No playback player after recording");
  console.log(`recording (~5 s): ${await page.textContent("#confidence")} | ${await page.textContent("#status")}`);

  if (modelFetches !== 1) throw new Error(`Model fetched ${modelFetches} times, expected 1`);
  if (errors.length) throw new Error(`Browser errors:\n${errors.join("\n")}`);
  const wasm = requested.filter((p) => p.endsWith(".wasm") || p.endsWith(".mjs"));
  console.log(`runtime files fetched: ${wasm.join(", ")}`);
  console.log("site test passed");
} finally {
  await browser.close();
  server.stop(true);
}
