/** Encode interleaved samples as a WAV file (for tests). */
export function encodeWav(
  samples: Float32Array,
  { sampleRate = 16000, channels = 1, format = "pcm16" }: {
    sampleRate?: number;
    channels?: number;
    format?: "pcm16" | "pcm24" | "float32";
  } = {},
): Uint8Array {
  const bytes = format === "pcm16" ? 2 : format === "pcm24" ? 3 : 4;
  const buf = new DataView(new ArrayBuffer(44 + samples.length * bytes));
  const str = (o: number, s: string) => [...s].forEach((c, i) => buf.setUint8(o + i, c.charCodeAt(0)));

  str(0, "RIFF");
  buf.setUint32(4, 36 + samples.length * bytes, true);
  str(8, "WAVE");
  str(12, "fmt ");
  buf.setUint32(16, 16, true);
  buf.setUint16(20, format === "float32" ? 3 : 1, true);
  buf.setUint16(22, channels, true);
  buf.setUint32(24, sampleRate, true);
  buf.setUint32(28, sampleRate * channels * bytes, true);
  buf.setUint16(32, channels * bytes, true);
  buf.setUint16(34, bytes * 8, true);
  str(36, "data");
  buf.setUint32(40, samples.length * bytes, true);

  samples.forEach((s, i) => {
    const o = 44 + i * bytes;
    if (format === "float32") buf.setFloat32(o, s, true);
    else if (format === "pcm16") buf.setInt16(o, Math.round(s * 32767), true);
    else {
      const v = Math.round(s * 8388607);
      buf.setUint8(o, v & 0xff);
      buf.setUint8(o + 1, (v >> 8) & 0xff);
      buf.setInt8(o + 2, v >> 16);
    }
  });
  return new Uint8Array(buf.buffer);
}

export function sine(freq: number, seconds: number, sampleRate: number): Float32Array {
  return Float32Array.from({ length: Math.round(seconds * sampleRate) }, (_, i) =>
    0.5 * Math.sin((2 * Math.PI * freq * i) / sampleRate),
  );
}
