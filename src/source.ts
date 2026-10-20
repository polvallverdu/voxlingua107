/**
 * The ONNX export of speechbrain/lang-id-voxlingua107-ecapa (FP32, raw 16 kHz audio in,
 * feature extraction inside the graph). Downloads are verified against this hash no matter
 * which URL they come from, so a mirror can't serve a different file.
 */
export const DEFAULT_MODEL = {
  sha256: "e2c3c3da39b99e3f9196d15fceef6a65f702320038bbc08813a4f21280255ce8",
  bytes: 86657056,
  urls: [
    "https://huggingface.co/polvallverdu/voxlingua107-onnx/resolve/80978d1e54491a9e05d4512e41b54626b87803f2/voxlingua107.onnx",
    "https://huggingface.co/christopherthompson81/voxlingua107-lid-onnx/resolve/e02e1da805ae49635fe1aa7913c3f1e7f5f5fde6/voxlingua107.onnx",
  ],
} as const;
