/** Sample rate the model expects. */
export const SAMPLE_RATE = 16000;

export interface DecodedAudio {
  /** Mono samples in [-1, 1]. */
  samples: Float32Array;
  sampleRate: number;
}

/** Convert 16-bit PCM to float samples in [-1, 1]. */
export function int16ToFloat32(pcm: Int16Array): Float32Array {
  const out = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) out[i] = pcm[i]! / 32768;
  return out;
}

/**
 * Decode a WAV file (PCM 8/16/24/32-bit or float32/64, any channel count) to mono float samples.
 * Channels are averaged. For other formats (mp3, opus, ...) decode with ffmpeg first.
 */
export function decodeWav(data: ArrayBuffer | ArrayBufferView): DecodedAudio {
  const view =
    data instanceof ArrayBuffer
      ? new DataView(data)
      : new DataView(data.buffer, data.byteOffset, data.byteLength);
  const tag = (o: number) =>
    String.fromCharCode(view.getUint8(o), view.getUint8(o + 1), view.getUint8(o + 2), view.getUint8(o + 3));

  if (view.byteLength < 12 || tag(0) !== "RIFF" || tag(8) !== "WAVE") {
    throw new Error("Not a RIFF/WAVE file");
  }

  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bits = 0;
  let dataOffset = -1;
  let dataLength = 0;

  for (let o = 12; o + 8 <= view.byteLength; ) {
    const id = tag(o);
    const size = view.getUint32(o + 4, true);
    const body = o + 8;
    if (id === "fmt ") {
      format = view.getUint16(body, true);
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      bits = view.getUint16(body + 14, true);
      // WAVE_FORMAT_EXTENSIBLE: the real format is the first 2 bytes of the subformat GUID.
      if (format === 0xfffe && size >= 26) format = view.getUint16(body + 24, true);
    } else if (id === "data") {
      dataOffset = body;
      dataLength = Math.min(size, view.byteLength - body);
      break;
    }
    o = body + size + (size & 1);
  }

  if (!channels || !sampleRate) throw new Error("WAV file has no fmt chunk");
  if (dataOffset < 0) throw new Error("WAV file has no data chunk");

  const read = sampleReader(view, format, bits);
  const bytesPerSample = bits / 8;
  const frames = Math.floor(dataLength / (bytesPerSample * channels));
  const samples = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) sum += read(dataOffset + (f * channels + c) * bytesPerSample);
    samples[f] = sum / channels;
  }
  return { samples, sampleRate };
}

function sampleReader(view: DataView, format: number, bits: number): (offset: number) => number {
  if (format === 1) {
    if (bits === 8) return (o) => (view.getUint8(o) - 128) / 128;
    if (bits === 16) return (o) => view.getInt16(o, true) / 32768;
    if (bits === 24)
      return (o) => ((view.getUint8(o) | (view.getUint8(o + 1) << 8) | (view.getInt8(o + 2) << 16)) / 8388608);
    if (bits === 32) return (o) => view.getInt32(o, true) / 2147483648;
  }
  if (format === 3) {
    if (bits === 32) return (o) => view.getFloat32(o, true);
    if (bits === 64) return (o) => view.getFloat64(o, true);
  }
  throw new Error(`Unsupported WAV encoding (format ${format}, ${bits}-bit)`);
}

/**
 * Resample with a Hann-windowed sinc filter (band-limited, so downsampling doesn't alias).
 * Returns the input unchanged when the rates match.
 */
export function resample(input: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return input;
  if (!(fromRate > 0 && toRate > 0)) throw new Error(`Invalid sample rates ${fromRate} -> ${toRate}`);

  const ratio = toRate / fromRate;
  const scale = Math.min(1, ratio); // lowpass at the lower Nyquist
  const zeros = 16; // filter half-width in zero crossings
  const halfWidth = zeros / scale; // in input samples
  const out = new Float32Array(Math.round(input.length * ratio));

  for (let i = 0; i < out.length; i++) {
    const center = i / ratio;
    const lo = Math.max(0, Math.ceil(center - halfWidth));
    const hi = Math.min(input.length - 1, Math.floor(center + halfWidth));
    let acc = 0;
    for (let j = lo; j <= hi; j++) {
      const x = (j - center) * scale;
      const sinc = x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x);
      const window = 0.5 + 0.5 * Math.cos((Math.PI * x) / zeros);
      acc += input[j]! * sinc * window;
    }
    out[i] = acc * scale;
  }
  return out;
}
