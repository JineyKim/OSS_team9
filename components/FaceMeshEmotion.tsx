"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

declare global {
  interface Window {
    FaceMesh: any;
    drawConnectors: any;
    drawLandmarks: any;
    FACEMESH_TESSELATION: any;
  }
}

//슬픔, 기쁨, 분노, 불안, 당황, 상처, 중립
const EMOTION_LABELS = [
  "슬픔",
  "기쁨",
  "화남",
  "불안",
  "당황",
  "상처",
  "중립",
];

function softmax(arr: number[]): number[] {
  const m = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / s);
}

// -----------------------
// FaceMesh → 1434 vector
// -----------------------
function faceLandmarksToVec1434(lm: any[]): Float32Array {
  const v = new Float32Array(478 * 3);
  for (let i = 0; i < 478; i++) {
    v[i * 3 + 0] = lm[i].x;
    v[i * 3 + 1] = lm[i].y;
    v[i * 3 + 2] = lm[i].z;
  }
  return v;
}

// -----------------------
// Apply StandardScaler
// -----------------------
function normalize(vec: Float32Array, mean: number[], scale: number[]) {
  const out = new Float32Array(1434);
  for (let i = 0; i < 1434; i++) {
    out[i] = (vec[i] - mean[i]) / scale[i];
  }
  return out;
}

// -----------------------
// ONNX inference
// -----------------------
async function runEmotionModel(
  session: ort.InferenceSession,
  normVec: Float32Array
) {
  const inputTensor = new ort.Tensor("float32", normVec, [1, 1434]);
  const output = await session.run({ input: inputTensor });
  const key = Object.keys(output)[0];

  const raw = Array.from(output[key].data as Float32Array);
  const probs = softmax(raw);
  const idx = probs.indexOf(Math.max(...probs));

  return { probs, idx, label: EMOTION_LABELS[idx] };
}

// -----------------------
// MAIN COMPONENT
// -----------------------
export default function FaceMeshEmotion() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [emotion, setEmotion] = useState("…");

  useEffect(() => {
    let fm: any = null;
    let session: ort.InferenceSession | null = null;
    let scaler: { mean: number[]; scale: number[] } | null = null;

    const FPS_INTERVAL = 1000 / 15;
    let lastFrame = 0;

    async function loadScript(src: string) {
      return new Promise<void>((resolve, reject) => {
        const s = document.createElement("script");
        s.src = src;
        s.onload = () => resolve();
        s.onerror = () => reject(`Failed to load: ${src}`);
        document.body.appendChild(s);
      });
    }

    async function init() {
      console.log("🔥 Initialize");

      // -------------------------
      // Load MediaPipe vendor
      // -------------------------
      await loadScript("/vendor/face_mesh/drawing_utils.js");
      await loadScript("/vendor/face_mesh/face_mesh.js");

      const { FaceMesh, drawConnectors, drawLandmarks, FACEMESH_TESSELATION } =
        window;

      // -------------------------
      // Load scaler.json
      // -------------------------
      console.log("📦 Loading scaler.json...");
      const response = await fetch("/model/scaler.json");
      scaler = await response.json();
      console.log("✔ scaler.json loaded");

      // -------------------------
      // Load emotion.onnx
      // -------------------------
      console.log("📦 Loading ONNX model…");
      session = await ort.InferenceSession.create("/model/emotion.onnx", {
        executionProviders: ["webgpu", "wasm"],
      });
      console.log("✔ ONNX ready");

      // -------------------------
      // Setup FaceMesh
      // -------------------------
      fm = new FaceMesh({
        locateFile: (file: string) => `/vendor/face_mesh/${file}`,
      });

      fm.setOptions({
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
        selfieMode: true,
        maxContinuousChecks: 1,
        useCpuInference: true   // 중요
      });

      fm.onResults(async (results: any) => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.image) {
          ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
        }

        if (!results.multiFaceLandmarks?.length) return;
        const lm = results.multiFaceLandmarks[0];

        drawConnectors(ctx, lm, FACEMESH_TESSELATION, {
          color: "#0f0",
          lineWidth: 1,
        });

        drawLandmarks(ctx, lm, {
          color: "#0ff",
          lineWidth: 1,
          radius: 0.4,    // 점 크기 줄임
        });

        if (session && scaler) {
          const vec = faceLandmarksToVec1434(lm);
          const norm = normalize(vec, scaler.mean, scaler.scale);
          const out = await runEmotionModel(session, norm);
          setEmotion(out.label);
        }
      });

      // -------------------------
      // Webcam
      // -------------------------
      const videoElem = videoRef.current!;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user", // 🔥 필수
          width: 640,
          height: 480,
        }
      });
      videoElem.srcObject = stream;
      await videoElem.play();

      // -------------------------
      // Render Loop
      // -------------------------
      async function loop() {
        const now = performance.now();
        if (now - lastFrame >= FPS_INTERVAL) {
          lastFrame = now;

          if (videoElem.readyState >= 2) {
            const vw = videoElem.videoWidth;
            const vh = videoElem.videoHeight;

            // 1) 가장 짧은 길이
            const size = Math.min(vw, vh);

            // 2) 가운데 크롭 좌표 계산
            const sx = (vw - size) / 2;  // crop start X
            const sy = (vh - size) / 2;  // crop start Y

            // 3) 캔버스도 정사각형 유지
            const canvas = canvasRef.current;
            const ctx = canvas?.getContext("2d");
            if (!canvas || !ctx) return;

            // 캔버스 크기를 정사각형으로
            canvas.width = size;
            canvas.height = size;

            // 4) crop한 정사각형 영상 그리기
            ctx.drawImage(
              videoElem,
              sx, sy, size, size,      // source crop
              0, 0, size, size         // destination
            );

            // 5) FaceMesh에 정사각형 frame 전달
            await fm.send({ image: canvas });
          }
        }
        requestAnimationFrame(loop);
      }

      loop();
    }

    init();
  }, []);

  return (
    <div style={{ textAlign: "center" }}>
      <h2>FaceMesh + Emotion</h2>
      <p style={{ fontSize: 22 }}>
        Emotion: <b>{emotion}</b>
      </p>

      <video ref={videoRef} style={{ display: "none" }} />

      <canvas
        ref={canvasRef}
        style={{
          width: "100vw",    // 화면 꽉 채움
          height: "100vw",   // 항상 정사각형
          objectFit: "cover",
        }}
      />
    </div>
  );
}
