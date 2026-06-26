/**
 * constants.ts
 * Valores compartidos entre el hilo principal, los componentes y el worker.
 * Centralizar aquí evita magic numbers dispersos por el código.
 */

/** Tamaño del lado cuadrado de entrada del modelo (px). YOLOv8n usa 640. */
export const MODEL_INPUT_SIZE = 640 as const;

/**
 * Modo del modelo activo.
 * - 'coco'  → YOLOv8n genérico (80 clases COCO)
 * - 'grape' → Modelo entrenado para detección de bayas de uva
 * Se configura en build-time con NEXT_PUBLIC_MODEL_MODE en Vercel.
 */
export const MODEL_MODE =
  (process.env.NEXT_PUBLIC_MODEL_MODE ?? 'coco') as 'coco' | 'grape';

/**
 * URL del modelo ONNX.
 * - En desarrollo: se sirve desde /public/models/ (local)
 * - En producción: Vercel Blob (CDN global, se configura con NEXT_PUBLIC_MODEL_URL)
 */
export const MODEL_URL =
  process.env.NEXT_PUBLIC_MODEL_URL ?? '/models/yolov8n.onnx';

/**
 * Umbral mínimo de confianza para mostrar una detección.
 * Grape usa 0.25 porque el modelo fue entrenado solo 10 epochs → scores más bajos.
 */
export const CONF_THRESHOLD = MODEL_MODE === 'grape' ? 0.25 : 0.40;

/** Umbral de IoU para Non-Maximum Suppression. */
export const IOU_THRESHOLD = 0.45 as const;

/** Número de clases COCO 2017 que YOLOv8n fue entrenado con. */
export const NUM_CLASSES = MODEL_MODE === 'grape' ? 1 : 80;

/**
 * Etiquetas COCO en orden de índice.
 * Fuente: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
 */
export const COCO_LABELS: readonly string[] = [
  'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
  'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
  'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
  'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
  'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
  'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
  'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
  'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
  'remote','keyboard','cell phone','microwave','oven','toaster','sink',
  'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
  'toothbrush',
] as const;

/**
 * Etiquetas del modelo de uvas (clases de grape-detection-xcemr en Roboflow).
 * Dataset: 10.6k imágenes de viñedos.
 * ⚠️  Si el script imprime clases distintas, actualiza este array en el mismo orden.
 */
export const GRAPE_LABELS: readonly string[] = [
  'Racimo',  // grape_cluster
] as const;

/**
 * Etiquetas activas según MODEL_MODE.
 * Úsalo en Detector.tsx en lugar de COCO_LABELS directamente.
 */
export const LABELS: readonly string[] =
  MODEL_MODE === 'grape' ? GRAPE_LABELS : COCO_LABELS;

/**
 * Paleta de colores por clase (HSL cycling para distinguir visualmente).
 * Se genera dinámicamente en base al classId para no hardcodear 80 colores.
 */
export function classColor(classId: number): string {
  const hue = (classId * 137.5) % 360; // golden angle para distribución uniforme
  return `hsl(${hue}, 90%, 55%)`;
}

// ── Tipos de mensajes del Worker ───────────────────────────────────────────

export interface Detection {
  x1: number;      // [0, 1] relativo al modelo
  y1: number;
  x2: number;
  y2: number;
  score: number;   // [0, 1]
  classId: number;
}

export type WorkerInMessage =
  | { type: 'INIT'; modelUrl: string }
  | { type: 'INFER'; bitmap: ImageBitmap }; // bitmap es transferible

export type WorkerOutMessage =
  | { type: 'READY'; backend: string }
  | { type: 'RESULT'; detections: Detection[]; inferMs: number }
  | { type: 'ERROR'; message: string }
  | { type: 'LOG'; text: string };
