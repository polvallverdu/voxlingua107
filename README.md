# voxlingua107

Detect which language someone is speaking (107 languages) from Node.js or Bun. It runs on CPU and doesn't need Python.

It runs SpeechBrain's [VoxLingua107 ECAPA-TDNN](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) model, exported to ONNX, on [ONNX Runtime](https://onnxruntime.ai). The test suite checks every result against the original PyTorch model.

```ts
import { LanguageIdentifier } from "voxlingua107";

const lid = await LanguageIdentifier.load(); // downloads the model once (~87 MB)
const result = await lid.identify(samples); // Float32Array or Int16Array, 16 kHz mono

result.code; // "th"
result.name; // "Thai"
result.probability; // 0.989
```

## Install

```sh
bun add voxlingua107
# or
npm install voxlingua107
```

Requires Node.js 20+ or Bun.

## Usage

### From a WAV file

```ts
import { readFile } from "node:fs/promises";
import { LanguageIdentifier, decodeWav } from "voxlingua107";

const lid = await LanguageIdentifier.load();
const { samples, sampleRate } = decodeWav(await readFile("speech.wav"));

const result = await lid.identify(samples, { sampleRate }); // resampled to 16 kHz for you
console.log(result.top); // the 5 most likely languages, best first
```

`decodeWav` handles 8/16/24/32-bit PCM and float WAVs with any channel count; channels are mixed down to mono. For mp3, opus and other formats, decode with ffmpeg first:

```sh
ffmpeg -i input.mp3 -ac 1 -ar 16000 -f f32le pipe:1   # raw Float32Array samples
```

### From a stream (e.g. Pipecat or WebRTC frames)

Audio frames are usually 16-bit PCM. Pass them directly and give the sample rate:

```ts
const pcm = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
const result = await lid.identify(pcm, { sampleRate: 24000 });
```

Collect a few seconds of speech before calling it (see [Accuracy](#accuracy)).

### Only consider some languages

When you know the possible languages, pass `candidates`. The probabilities are renormalized over that set:

```ts
const result = await lid.identify(samples, { candidates: ["en", "es", "ca"] });
```

### Typed language codes

`result.code` and `candidates` use the `LanguageCode` type, a union of all 107 codes (`"en" | "es" | "haw" | ...`), so a typo is a compile error. Use `isLanguageCode()` to check strings that arrive at runtime:

```ts
import { isLanguageCode, type LanguageCode } from "voxlingua107";

const voices: Partial<Record<LanguageCode, string>> = { en: "alloy", es: "nova" };

const lang = process.env.DEFAULT_LANG ?? "en";
if (!isLanguageCode(lang)) throw new Error(`Unsupported language: ${lang}`);
```

## API

### `LanguageIdentifier.load(options?)`

Loads the model and returns a `LanguageIdentifier`. Load it once and reuse it. Concurrent calls are safe.

| Option           | Default                                    | Description                                                     |
| ---------------- | ------------------------------------------ | --------------------------------------------------------------- |
| `cacheDir`       | see [Model download](#model-download-and-caching) | Where the model is cached.                               |
| `modelPath`      |                                            | Use this local `.onnx` file. Nothing is downloaded.             |
| `download`       | `true`                                     | `false` throws instead of downloading when the model isn't cached. |
| `urls`           | HF mirror, then the original export        | Download URLs, tried in order.                                  |
| `sha256`         | the pinned model hash                      | Expected hash of the download. `null` skips the check.          |
| `onProgress`     |                                            | `(loaded, total) => void` download progress.                    |
| `sessionOptions` |                                            | Passed to ONNX Runtime (e.g. `{ intraOpNumThreads: 2 }`).       |

### `lid.identify(audio, options?)` → `Promise<IdentifyResult>`

`audio` is a mono `Float32Array` (values in [-1, 1]) or `Int16Array`.

| Option       | Default | Description                                     |
| ------------ | ------- | ----------------------------------------------- |
| `sampleRate` | `16000` | Sample rate of `audio`. Resampled if not 16 kHz. |
| `topK`       | `5`     | Number of predictions in `top`.                 |
| `candidates` | all     | Restrict to these language codes.               |

Returns `{ code, name, probability, top }`, where `top` is `[{ code, name, probability }, ...]` sorted best first.

### Also exported

- `lid.embed(audio, options?)`: the 256-dim language embedding (`Float32Array`).
- `lid.run(audio, options?)`: raw outputs, `{ logits, embedding }`. `logits` are log-probabilities in `LANGUAGES` order.
- `lid.dispose()`: frees the ONNX Runtime session.
- `LANGUAGES`: all 107 `{ code, name }` entries, in the model's output order.
- `LanguageCode`: union type of the 107 codes. `isLanguageCode(value)` narrows a `string` to it.
- `decodeWav`, `resample`, `int16ToFloat32`: the audio helpers used above.
- `resolveModel(options?)`: downloads/locates the model and returns its path, without loading it.

## Model download and caching

On first use, the model (~87 MB) is downloaded and stored in the cache. The first directory that's set wins:

1. the `cacheDir` option
2. `$VOXLINGUA107_CACHE_DIR`
3. `$XDG_CACHE_HOME/voxlingua107`
4. `~/.cache/voxlingua107`

Every download is checked against a pinned SHA-256 before it's used, whichever URL it came from. The file is downloaded from [`polvallverdu/voxlingua107-onnx`](https://huggingface.co/polvallverdu/voxlingua107-onnx), a pinned mirror, and if that fails, from the [original ONNX export](https://huggingface.co/christopherthompson81/voxlingua107-lid-onnx).

**Docker, serverless and offline setups.** Download the model at build time and turn off runtime downloads:

```dockerfile
ENV VOXLINGUA107_CACHE_DIR=/models
RUN npx voxlingua107 download
```

```ts
const lid = await LanguageIdentifier.load({ download: false });
```

You can also ship the file yourself and pass `modelPath: "/path/to/voxlingua107.onnx"`.

## CLI

```sh
npx voxlingua107 identify speech.wav          # other formats need ffmpeg on PATH
npx voxlingua107 identify speech.mp3 --top 3
npx voxlingua107 download --cache-dir ./models
```

```
th   Thai               98.9%
lo   Lao                1.1%
la   Latin              0.0%
```

## Accuracy

The model is SpeechBrain's VoxLingua107 ECAPA-TDNN. Its authors report a 6.7% error rate on the VoxLingua107 dev set. This package runs that model unchanged, and the tests check it against the original:

| Test clip (from [`scripts/generate-reference.py`](scripts/generate-reference.py)) | Max probability difference vs PyTorch | Same top-1 |
| --- | --- | --- |
| Upstream example, Thai, 13 s | 1.8e-6 | ✅ |
| Same, first 3 s | 3.8e-6 | ✅ |
| Same, 1 s | 5.8e-6 | ✅ |
| Random noise, 4 s | 3.1e-7 | ✅ |

Practical notes:

- **Give it a few seconds of speech.** Under about 3 s, results get noticeably less reliable. Beyond about 30 s, accuracy stops improving.
- It was trained on YouTube audio. It works best on similar speech and worse on strong foreign accents, children's speech, and some smaller languages.
- Hebrew and Javanese are returned as `he` and `jv`. The upstream labels use the obsolete codes `iw` and `jw` ([speechbrain#2396](https://github.com/speechbrain/speechbrain/issues/2396)).

## Performance

CPU, single call, Apple M4 Pro: about 25 ms for 3 s of audio, 50 ms for 10 s, and 170 ms for 30 s. Loading the model takes well under a second once it's cached.

## Development

```sh
bun install
bun test             # unit + model parity tests (downloads the model on first run)
bun run typecheck
bun run test:node    # builds and runs the package under plain Node
```

To regenerate the PyTorch reference outputs, see the comment at the top of [`scripts/generate-reference.py`](scripts/generate-reference.py). It needs [uv](https://docs.astral.sh/uv/).

### Releasing

Go to **Actions → Release → Run workflow** and choose `patch`, `minor`, `major` or `prerelease`. The workflow runs the tests, bumps the version, publishes to npm with provenance, pushes the tag and creates a GitHub release.

One-time setup: on npmjs.com, open the package's **Settings → Trusted Publisher** and add GitHub Actions with repository `polvallverdu/voxlingua107` and workflow `release.yml`. npm can only configure that for a package that already exists. Until then, either publish `0.1.0` once by hand (`npm publish`), or add an `NPM_TOKEN` secret to the repo.

## Credits and license

Apache-2.0.

- Model: Jörgen Valk and Tanel Alumäe, [VoxLingua107](https://arxiv.org/abs/2011.12998), trained with [SpeechBrain](https://github.com/speechbrain/speechbrain).
- ONNX export: [Chris Thompson](https://github.com/christopherthompson81) ([conversion scripts](https://github.com/christopherthompson81/vernacula/tree/main/scripts/voxlingua107_export)).
- Test clip `udhr_th.wav` comes from the [upstream model repository](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa).
