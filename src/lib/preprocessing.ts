/**
 * preprocessing.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Pipeline de preprocesamiento y postprocesamiento de imagen para YOLO ONNX.
 * Este módulo se importa ÚNICAMENTE desde el Web Worker (contexto WorkerGlobalScope).
 *
 * Flujo:
 *   ImageBitmap → OffscreenCanvas (resize GPU) → getImageData → Float32Array CHW
 * ─────────────────────────────────────────────────────────────────────────────
 */

import {
  MODEL_INPUT_SIZE,
  CONF_THRESHOLD,
  IOU_THRESHOLD,
  NUM_CLASSES,
  type Detection,
} from './constants';

// ── OffscreenCanvas reutilizable ──────────────────────────────────────────────
// Se inicializa una sola vez por ciclo de vida del worker.
// Evitar crear nuevos canvas por frame es crítico para no presionar el GC.
let _canvas: OffscreenCanvas | null = null;
let _ctx: OffscreenCanvasRenderingContext2D | null = null;

function getCtx(size: number): OffscreenCanvasRenderingContext2D {
  if (!_canvas || _canvas.width !== size) {
    _canvas = new OffscreenCanvas(size, size);
    // willReadFrequently: el browser mantiene una copia en CPU-accessible memory,
    // evitando readbacks costosos desde GPU en cada getImageData().
    _ctx = _canvas.getContext('2d', { willReadFrequently: true, alpha: false })!;
  }
  return _ctx!;
}

// ── Preprocesamiento ──────────────────────────────────────────────────────────

/**
 * Convierte un ImageBitmap al tensor Float32Array [1, 3, H, W] (NCHW) que YOLO espera.
 *
 * @param bitmap    Frame de cámara transferido desde el hilo principal.
 * @param inputSize Tamaño cuadrado de entrada del modelo (default MODEL_INPUT_SIZE).
 * @returns         Tensor Float32Array listo para ort.Tensor.
 */
export function bitmapToTensor(
  bitmap: ImageBitmap,
  inputSize = MODEL_INPUT_SIZE
): Float32Array {
  const ctx = getCtx(inputSize);

  // 1. RESIZE: drawImage escala el bitmap usando el compositor del worker.
  //    En browsers con WebGPU, esto puede ocurrir parcialmente en GPU.
  ctx.drawImage(bitmap, 0, 0, inputSize, inputSize);

  // Liberamos el bitmap inmediatamente; ya no lo necesitamos y ocupa memoria GPU.
  bitmap.close();

  // 2. READBACK: obtener los píxeles en CPU.
  //    Formato: Uint8ClampedArray interleaved [R, G, B, A, R, G, B, A, ...]
  const { data } = ctx.getImageData(0, 0, inputSize, inputSize);

  // 3. NORMALIZACIÓN + CONVERSIÓN A CHW:
  //    YOLO espera el tensor en formato planar (todos los R, luego G, luego B)
  //    con valores en [0.0, 1.0].
  const N = inputSize * inputSize; // número total de píxeles
  const tensor = new Float32Array(3 * N);

  // Pre-calculamos offsets para evitar multiplicaciones dentro del loop
  const rPlane = 0;
  const gPlane = N;
  const bPlane = N * 2;

  for (let i = 0; i < N; i++) {
    const s = i << 2; // i * 4 (RGBA stride)
    tensor[rPlane + i] = data[s]     / 255.0;
    tensor[gPlane + i] = data[s + 1] / 255.0;
    tensor[bPlane + i] = data[s + 2] / 255.0;
    // Canal Alpha ignorado: YOLO no lo usa
  }

  return tensor;
}

// ── Postprocesamiento ─────────────────────────────────────────────────────────

/**
 * Decodifica la salida cruda de YOLOv8 y aplica NMS.
 *
 * YOLOv8 ONNX exportado produce un tensor de shape [1, 84, 8400]:
 *   - 84 = 4 (cx, cy, w, h) + 80 clases
 *   - 8400 = cantidad de anchors candidatos
 *
 * NOTA: la salida ya viene con coordenadas en píxeles del espacio del modelo
 * (0–640), NO normalizadas. Las normalizamos a [0, 1] dividiendo por inputSize.
 *
 * @param rawOutput   Float32Array con la salida del modelo (shape aplanado).
 * @param numClasses  Número de clases (default NUM_CLASSES).
 * @param confThresh  Umbral de confianza (default CONF_THRESHOLD).
 * @param iouThresh   Umbral IoU para NMS (default IOU_THRESHOLD).
 * @param inputSize   Tamaño de entrada del modelo para normalizar coords.
 * @returns           Array de detecciones filtradas y ordenadas por score.
 */
export function decodeAndNMS(
  rawOutput: Float32Array,
  numClasses = NUM_CLASSES,
  confThresh = CONF_THRESHOLD,
  iouThresh  = IOU_THRESHOLD,
  inputSize  = MODEL_INPUT_SIZE
): Detection[] {
  // YOLOv8 ONNX output layout: [1, 84, 8400] → feature-first cuando se aplana.
  // rawOutput[feature * numAnchors + anchor]
  //   features 0-3  → cx, cy, w, h  (en píxeles del espacio del modelo)
  //   features 4-N  → scores de clase
  const numAnchors = 8400;

  // Auto-detectar numClasses desde el tamaño real del output tensor.
  // Más fiable que session.outputMetadata (puede no estar disponible en ORT Web).
  // rawOutput.length = (4 + numClasses) * 8400 → numClasses = length/8400 - 4
  const inferredClasses = Math.round(rawOutput.length / numAnchors) - 4;
  if (inferredClasses > 0 && inferredClasses !== numClasses) {
    console.info(`[NMS] numClasses ajustado: ${numClasses} → ${inferredClasses}`);
    numClasses = inferredClasses;
  }

  const candidates: Detection[] = [];

  for (let a = 0; a < numAnchors; a++) {
    // Coordenadas: cada feature ocupa un plano de 8400 valores
    const cx = rawOutput[0 * numAnchors + a];
    const cy = rawOutput[1 * numAnchors + a];
    const bw = rawOutput[2 * numAnchors + a];
    const bh = rawOutput[3 * numAnchors + a];

    // Buscar la clase con mayor score
    let maxScore = 0;
    let classId  = 0;
    for (let c = 0; c < numClasses; c++) {
      const s = rawOutput[(4 + c) * numAnchors + a];
      if (s > maxScore) { maxScore = s; classId = c; }
    }

    if (maxScore < confThresh) continue;

    // Convertir cx,cy,w,h → x1,y1,x2,y2 y normalizar a [0,1]
    candidates.push({
      x1: Math.max(0, (cx - bw * 0.5) / inputSize),
      y1: Math.max(0, (cy - bh * 0.5) / inputSize),
      x2: Math.min(1, (cx + bw * 0.5) / inputSize),
      y2: Math.min(1, (cy + bh * 0.5) / inputSize),
      score: maxScore,
      classId,
    });
  }

  // NMS: ordenar por score desc, suprimir solapamientos por clase
  candidates.sort((a, b) => b.score - a.score);

  const suppressed = new Uint8Array(candidates.length);
  const kept: Detection[] = [];

  for (let i = 0; i < candidates.length; i++) {
    if (suppressed[i]) continue;
    kept.push(candidates[i]);
    for (let j = i + 1; j < candidates.length; j++) {
      if (suppressed[j]) continue;
      if (candidates[j].classId !== candidates[i].classId) continue;
      if (computeIoU(candidates[i], candidates[j]) > iouThresh) {
        suppressed[j] = 1;
      }
    }
  }

  return kept;
}

/** Intersection over Union entre dos bounding boxes normalizados. */
function computeIoU(a: Detection, b: Detection): number {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);

  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;

  if (inter === 0) return 0;

  const aArea = (a.x2 - a.x1) * (a.y2 - a.y1);
  const bArea = (b.x2 - b.x1) * (b.y2 - b.y1);

  return inter / (aArea + bArea - inter);
}
