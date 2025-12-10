"use client";

import * as ort from "onnxruntime-web";

export interface EmotionProbs {
  probs: Float32Array;
  idx: number;
  label: string;
}

const EMO_LABELS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"];

export async function loadEmotionOnnxModel(): Promise<ort.InferenceSession> {
  return await ort.InferenceSession.create("/model/emotion.onnx", {
    executionProviders: ["wasm"],
  });
}

export async function runEmotionModel(
  session: ort.InferenceSession,
  input: Float32Array
): Promise<EmotionProbs> {
  const tensor = new ort.Tensor("float32", input, [1, 1, 64, 64]);
  const out = await session.run({ input: tensor });
  const arr = out.output.data as Float32Array;

  let idx = 0;
  let maxVal = -999;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > maxVal) {
      idx = i;
      maxVal = arr[i];
    }
  }

  return {
    probs: arr,
    idx,
    label: EMO_LABELS[idx],
  };
}
