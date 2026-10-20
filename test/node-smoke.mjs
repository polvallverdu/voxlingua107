// Runs the built package under plain Node (no Bun, no TypeScript) to catch ESM/packaging issues.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { LanguageIdentifier, decodeWav } from "voxlingua107";

const { samples } = decodeWav(await readFile(new URL("./fixtures/th_full.wav", import.meta.url)));
const lid = await LanguageIdentifier.load();
const result = await lid.identify(samples);
await lid.dispose();

assert.equal(result.code, "th");
assert.ok(result.probability > 0.98, `probability ${result.probability}`);
console.log(`node ${process.version}: ok (${result.name}, ${(result.probability * 100).toFixed(1)}%)`);
