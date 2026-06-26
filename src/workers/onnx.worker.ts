/**
 * onnx.worker.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Web Worker dedicado a la inferencia ONNX Runtime Web.
 *
 * Por qué un Worker dedicado:
 *  - La inferencia puede tardar 20–150 ms según dispositivo y backend.
 *    En el hilo principal eso bloquea el compositor y congela la UI.
 *  - El Worker tiene su propio event loop; el hilo principal sigue pintando a 60 FPS.
 *  - WebGPU/WebGL tienen sus propios contextos y no compiten con el DOM.
 *
 * Protocolo de mensajes:
 *  IN  { type: 'INIT',  modelUrl: string }       → carga modelo
 *  IN  { type: 'INFER', bitmap: ImageBitmap }     → inferencia (bitmap transferido)
 *  OUT { type: 'READY', backend: string }
 *  OUT { type: 'RESULT', detections, inferMs }
 *  OUT { type: 'ERROR',  message: string }
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Nota para webpack/Next.js:
 *  Este archivo se importa en el cliente con:
 *    new Worker(new URL('./onnx.worker.ts', import.meta.url))
 *  Next.js 13+ con webpack 5 empaqueta el worker de forma automática.
 */

// Indicamos al TypeScript que este archivo corre en contexto Worker.
/// <reference lib="webworker" />

import * as ort from 'onnxruntime-web';
import { bitmapToTensor, decodeAndNMS } from '@/lib/preprocessing';
import { MODEL_INPUT_SIZE, type WorkerInMessage, type WorkerOutMessage } from '@/lib/constants';

// ── Estado del worker ─────────────────────────────────────────────────────────
let session:       ort.InferenceSession | null = null;
let inputName:     string = '';
let activeBackend: string = 'unknown';
let numClasses:    number = 80;

/** Envía un log al hilo principal para mostrarlo en el HUD. */
function postLog(text: string): void {
  console.info(text);
  self.postMessage({ type: 'LOG', text });
}

// ─────────────────────────────────────────────────────────────────────────────
// CONFIGURACIÓN DEL ENTORNO ORT
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Configura las variables globales de ONNX Runtime antes de crear la sesión.
 *
 * Estrategia de backends (de más rápido a más compatible):
 *  webgpu → Acceso directo a GPU moderna (Chrome 113+, Android Chrome).
 *            ~3–15 ms en modelos ligeros; sin copia de datos CPU↔GPU en operaciones GPU-to-GPU.
 *  webgl  → Shaders GLSL, más compatible (Safari iOS 16+, Android WebView).
 *            ~10–40 ms; copia datos en cada inferencia.
 *  wasm   → CPU puro con SIMD. Siempre disponible.
 *            ~40–200 ms en móviles; multi-thread si hay SharedArrayBuffer.
 */
function configureOrtEnv(): void {
  // ── Rutas WASM ────────────────────────────────────────────────────────────
  // onnxruntime-web instala sus .wasm en node_modules/onnxruntime-web/dist/.
  // Next.js los copia a /_next/static/chunks/ durante el build.
  // La forma más segura de apuntarlos en un Worker es con la ruta del paquete.
  // Los .wasm se sirven desde /public/ort/ (copiados en next.config.ts)
  ort.env.wasm.wasmPaths = '/ort/';

  // ── Threads WASM ─────────────────────────────────────────────────────────
  // Multi-threading WASM requiere SharedArrayBuffer (headers COOP/COEP en next.config.ts).
  // Usamos todos los cores disponibles.
  ort.env.wasm.numThreads = (navigator as Navigator & { hardwareConcurrency?: number })
    .hardwareConcurrency ?? 4;

  // ── WebGPU power preference ───────────────────────────────────────────────
  // 'high-performance' solicita la GPU dedicada en laptops con GPU dual.
  // En móviles suele ignorarse (solo tienen una GPU) pero no causa errores.
  ort.env.webgpu = {
    powerPreference: 'high-performance',
  } as typeof ort.env.webgpu;
}

// ─────────────────────────────────────────────────────────────────────────────
// CARGA DEL MODELO
// ─────────────────────────────────────────────────────────────────────────────

async function initModel(modelUrl: string): Promise<void> {
  configureOrtEnv();

  // ── Execution Providers ───────────────────────────────────────────────────
  // ORT prueba cada provider en orden; si uno falla, pasa al siguiente.
  const executionProviders: ort.InferenceSession.ExecutionProviderConfig[] = [
    // WebGPU: el más rápido en hardware moderno.
    // preferredLayout NCHW porque YOLOv8 ONNX está exportado en ese formato.
    { name: 'webgpu', preferredLayout: 'NCHW' } as ort.InferenceSession.ExecutionProviderConfig,
    // WebGL: fallback GPU para dispositivos sin WebGPU.
    'webgl',
    // WASM CPU: garantiza que la PoC siempre corra, en cualquier dispositivo.
    'wasm',
  ];

  const options: ort.InferenceSession.SessionOptions = {
    executionProviders,

    // graphOptimizationLevel 'all': aplica constant folding, operator fusion,
    // y eliminación de nodos redundantes en tiempo de carga.
    // Aumenta el tiempo de init ~10-20% pero reduce la latencia de cada inferencia.
    graphOptimizationLevel: 'all',

    // Arenas de memoria pre-alocadas → menos malloc/free durante la inferencia.
    enableCpuMemArena: true,
    enableMemPattern:  true,

    // sequential: mejor para grafos sin mucho paralelismo de nodos (como YOLO).
    executionMode: 'sequential',

    // freeDimensionOverrides: fija dimensiones dinámicas del modelo (batch, h, w)
    // para que el compilador de operaciones genere código optimizado para este tamaño exacto.
    freeDimensionOverrides: {
      batch_size: 1,
      height:     MODEL_INPUT_SIZE,
      width:      MODEL_INPUT_SIZE,
    },
  };

  postLog(`📦 Cargando: ${modelUrl.split('/').pop()}`);
  const t0 = performance.now();

  session = await ort.InferenceSession.create(modelUrl, options);

  postLog(`✅ Modelo cargado en ${(performance.now() - t0).toFixed(0)} ms`);

  // ── Detectar backend activo ───────────────────────────────────────────────
  // No hay API pública estable; usamos heurística basada en la disponibilidad
  // de objetos globales dentro del Worker scope.
  if (typeof (self as unknown as Record<string, unknown>).GPUDevice !== 'undefined') {
    activeBackend = 'webgpu';
  } else if (typeof WebGLRenderingContext !== 'undefined') {
    activeBackend = 'webgl';
  } else {
    activeBackend = 'wasm';
  }

  // ── Inspeccionar metadatos del modelo ─────────────────────────────────────
  inputName = session.inputNames[0];

  // Inferir número de clases desde la forma del tensor de salida.
  // YOLOv8 COCO: output shape = [1, 84, 8400] → 84 - 4 = 80 clases.
  const outMeta = session.outputMetadata?.[session.outputNames[0]];
  if (outMeta?.dimensions) {
    numClasses = (outMeta.dimensions[1] as number) - 4;
  }

  postLog(`🔎 input: ${inputName} | clases: ${numClasses} | ${activeBackend}`);
}

// ─────────────────────────────────────────────────────────────────────────────
// CICLO DE INFERENCIA
// ─────────────────────────────────────────────────────────────────────────────

async function runInference(bitmap: ImageBitmap): Promise<{ inferMs: number }> {
  if (!session) throw new Error('Sesión no inicializada');

  // 1. PREPROCESAR: ImageBitmap → Float32Array CHW (CPU, ~1-3 ms en móvil)
  const tensorData = bitmapToTensor(bitmap);
  // bitmap ya se cerró dentro de bitmapToTensor → memoria liberada

  // 2. CREAR TENSOR ORT
  // ort.Tensor recibe el TypedArray directamente → no copia en el backend WASM.
  // Para WebGPU/WebGL, ORT hace el upload interno durante la inferencia.
  const inputTensor = new ort.Tensor('float32', tensorData, [
    1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,  // NCHW
  ]);

  // 3. INFERENCIA
  const t0 = performance.now();
  const outputs = await session.run({ [inputName]: inputTensor });
  const inferMs = performance.now() - t0;

  // 4. LIBERAR TENSOR DE ENTRADA explícitamente.
  // Algunos providers (WebGPU) mantienen buffers en GPU hasta que se llama dispose().
  // Llamarlo manualmente reduce la presión del GC de GPU.
  if ('dispose' in inputTensor) {
    (inputTensor as ort.Tensor & { dispose(): void }).dispose();
  }

  // 5. OBTENER SALIDA y DECODIFICAR
  const rawOutput = outputs[session.outputNames[0]].data as Float32Array;

  // Auto-detectar numClasses desde el tamaño real del tensor (solo en el primer frame)
  const inferredClasses = Math.round(rawOutput.length / 8400) - 4;
  if (inferredClasses > 0 && inferredClasses !== numClasses) {
    postLog(`🔧 clases: ${numClasses}→${inferredClasses} | output: ${rawOutput.length}`);
    numClasses = inferredClasses;
  }

  const detections = decodeAndNMS(rawOutput, numClasses);

  // Cada ~60 frames loguear el score máximo para diagnóstico
  if (!('_diagCount' in (self as unknown as Record<string, unknown>))) {
    (self as unknown as Record<string, unknown>)._diagCount = 0;
  }
  const diagCount = (self as unknown as Record<string, number>)._diagCount++;
  if (diagCount % 60 === 0) {
    let maxScore = 0;
    const numAnchors = 8400;
    for (let a = 0; a < numAnchors; a++) {
      for (let c = 0; c < numClasses; c++) {
        const s = rawOutput[(4 + c) * numAnchors + a];
        if (s > maxScore) maxScore = s;
      }
    }
    postLog(`📊 maxScore: ${(maxScore * 100).toFixed(1)}% | dets: ${detections.length}`);
  }

  return { inferMs, detections } as unknown as { inferMs: number };

  // NOTA: retornamos el objeto completo pero el tipo declara solo inferMs
  // para evitar re-declaración; el cast es intencional aquí.
}

// ─────────────────────────────────────────────────────────────────────────────
// MANEJADOR DE MENSAJES
// ─────────────────────────────────────────────────────────────────────────────

self.addEventListener('message', async (event: MessageEvent<WorkerInMessage>) => {
  const msg = event.data;

  try {
    if (msg.type === 'INIT') {
      await initModel(msg.modelUrl);
      const reply: WorkerOutMessage = { type: 'READY', backend: activeBackend };
      self.postMessage(reply);
      return;
    }

    if (msg.type === 'INFER') {
      // runInference recibe el bitmap (ya transferido, zero-copy)
      const result = await (runInference(msg.bitmap) as unknown as Promise<{
        detections: import('@/lib/constants').Detection[];
        inferMs: number;
      }>);

      const reply: WorkerOutMessage = {
        type: 'RESULT',
        detections: result.detections,
        inferMs:    result.inferMs,
      };
      // detections es un Array de objetos planos → se serializa por valor (barato).
      self.postMessage(reply);
    }

  } catch (err) {
    const reply: WorkerOutMessage = {
      type: 'ERROR',
      message: err instanceof Error ? err.message : String(err),
    };
    self.postMessage(reply);
  }
});
