/**
 * Detector.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * Componente cliente principal de la PoC.
 *
 * Responsabilidades:
 *  1. Acceder a la cámara trasera del dispositivo móvil.
 *  2. Instanciar el Web Worker de inferencia ONNX.
 *  3. Ejecutar el bucle de renderizado con requestAnimationFrame (rAF).
 *  4. Enviar frames al worker como ImageBitmap (transferible, zero-copy).
 *  5. Recibir las detecciones y dibujarlas sobre un <canvas> superpuesto.
 *  6. Actualizar el HUD con métricas de rendimiento.
 * ─────────────────────────────────────────────────────────────────────────────
 */
'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import { HUD } from './HUD';
import { MODEL_URL, LABELS, MODEL_MODE, classColor, type Detection, type WorkerOutMessage } from '@/lib/constants';

// ── Constantes de rendimiento ────────────────────────────────────────────────
/** Tamaño del cuadrado que enviamos al worker (puede ser menor que MODEL_INPUT_SIZE
 *  si queremos reducir el tiempo de captureBitmap en dispositivos lentos). */
const CAPTURE_SIZE = 640;

export function Detector() {
  // ── Refs (no re-renderizan el componente al cambiar) ─────────────────────
  const videoRef   = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const workerRef  = useRef<Worker | null>(null);

  // Controla si el worker está ocupado procesando un frame anterior.
  // Patrón "one-at-a-time": evita saturar el worker con más frames de los que puede procesar.
  const inferBusy  = useRef(false);

  // Referencia al ID del rAF para poder cancelarlo al desmontar.
  const rafId      = useRef<number>(0);

  // Última lista de detecciones recibida (se lee en el loop rAF para dibujar).
  const detections = useRef<Detection[]>([]);

  // ── Estado React (sí produce re-render, pero ocurre < 1 vez/segundo) ─────
  const [backend, setBackend]   = useState('');
  const [inferMs, setInferMs]   = useState<number | null>(null);
  const [fps,     setFps]       = useState<number | null>(null);
  const [count,   setCount]     = useState<number | null>(null);
  const [status,  setStatus]    = useState('Iniciando cámara…');
  const [logs,    setLogs]      = useState<string[]>([]);

  // Captura console.info para mostrarlo en pantalla
  const addLog = useCallback((msg: string) => {
    setLogs(prev => [...prev.slice(-6), msg]);
  }, []);

  // ── FPS counter ───────────────────────────────────────────────────────────
  const fpsFrames = useRef(0);
  const fpsTime   = useRef(performance.now());

  // ─────────────────────────────────────────────────────────────────────────
  // DIBUJO DE BOUNDING BOXES
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Dibuja las detecciones actuales sobre el canvas overlay.
   * Se llama en cada frame del loop rAF, incluso si no hay detecciones nuevas.
   * Usar requestAnimationFrame garantiza sincronización con el vsync del display.
   */
  const drawDetections = useCallback((canvas: HTMLCanvasElement) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    // Limpiar el frame anterior con clearRect (mucho más rápido que canvas.width = canvas.width)
    ctx.clearRect(0, 0, W, H);

    for (const det of detections.current) {
      const x  = det.x1 * W;
      const y  = det.y1 * H;
      const bw = (det.x2 - det.x1) * W;
      const bh = (det.y2 - det.y1) * H;

      const color = classColor(det.classId);
      const label = `${LABELS[det.classId] ?? det.classId} ${(det.score * 100).toFixed(0)}%`;

      // Caja
      ctx.strokeStyle = color;
      ctx.lineWidth   = 2;
      ctx.strokeRect(x, y, bw, bh);

      // Etiqueta de texto con fondo
      ctx.font         = 'bold 12px system-ui';
      const textW      = ctx.measureText(label).width + 8;
      const textH      = 18;
      ctx.fillStyle    = color;
      ctx.fillRect(x, y - textH, textW, textH);
      ctx.fillStyle    = '#000';
      ctx.fillText(label, x + 4, y - 4);
    }
  }, []);

  // ─────────────────────────────────────────────────────────────────────────
  // BUCLE rAF PRINCIPAL
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Loop de renderizado optimizado.
   *
   * Estrategia de envío de frames:
   *  - El rAF se dispara a ~60 Hz (o a la frecuencia del display del dispositivo).
   *  - Solo enviamos un frame al worker si `inferBusy` es false.
   *  - Esto implementa "backpressure": si el worker tarda 50 ms (20 FPS de inferencia),
   *    el hilo principal sigue dibujando a 60 FPS con las últimas detecciones conocidas.
   *  - Evita que se acumulen frames en la cola del worker (latencia percibida alta).
   */
  const startRenderLoop = useCallback((video: HTMLVideoElement, canvas: HTMLCanvasElement, worker: Worker) => {
    // Sincronizar el tamaño del canvas con el video al inicio
    canvas.width  = video.videoWidth  || CAPTURE_SIZE;
    canvas.height = video.videoHeight || CAPTURE_SIZE;

    const loop = () => {
      rafId.current = requestAnimationFrame(loop);

      // ── FPS counter ─────────────────────────────────────────────────────
      fpsFrames.current++;
      const now = performance.now();
      if (now - fpsTime.current >= 1000) {
        setFps(fpsFrames.current);
        fpsFrames.current = 0;
        fpsTime.current   = now;
      }

      // ── Dibujar detecciones previas ─────────────────────────────────────
      // Se dibuja cada frame para que el canvas no quede obsoleto.
      drawDetections(canvas);

      // ── Enviar frame al worker (solo si está libre) ─────────────────────
      if (!inferBusy.current && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        inferBusy.current = true;

        // createImageBitmap recorta/escala el frame de video al tamaño de captura.
        // El resultado es transferible (zero-copy al worker).
        // resizeQuality 'pixelated' es el más rápido; la calidad no afecta la precisión del modelo.
        createImageBitmap(video, {
          resizeWidth:   CAPTURE_SIZE,
          resizeHeight:  CAPTURE_SIZE,
          resizeQuality: 'pixelated',
        }).then((bitmap) => {
          // Transferimos el bitmap al worker sin copia de memoria.
          // Una vez transferido, este contexto ya no puede acceder al bitmap.
          worker.postMessage({ type: 'INFER', bitmap }, [bitmap]);
        }).catch(() => {
          // Si el frame no está listo (raro pero posible), simplemente lo saltamos.
          inferBusy.current = false;
        });
      }
    };

    rafId.current = requestAnimationFrame(loop);
  }, [drawDetections]);

  // ─────────────────────────────────────────────────────────────────────────
  // SETUP: CÁMARA + WORKER
  // ─────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    const video   = videoRef.current!;
    const canvas  = overlayRef.current!;
    let stream: MediaStream | null = null;
    let stopped   = false;

    // ── 1. Iniciar cámara trasera ──────────────────────────────────────────
    async function startCamera() {
      try {
        // Constraints optimizados para inferencia en tiempo real:
        // - facingMode: 'environment' → cámara trasera (más calidad en móviles)
        // - width/height ideales de 640 para que coincidan con el modelo
        // - frameRate 30 es suficiente; 60 no mejora la precisión del modelo
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode:  { ideal: 'environment' },
            width:       { ideal: CAPTURE_SIZE },
            height:      { ideal: CAPTURE_SIZE },
            frameRate:   { ideal: 30, max: 60 },
          },
          audio: false,
        });

        if (stopped) { stream.getTracks().forEach(t => t.stop()); return; }

        video.srcObject = stream;
        await video.play();
        setStatus('Cargando modelo ONNX…');
      } catch (err) {
        setStatus(`Error de cámara: ${(err as Error).message}`);
      }
    }

    // ── 2. Iniciar Worker ──────────────────────────────────────────────────
    // La sintaxis `new URL(…, import.meta.url)` es la forma estándar de
    // referenciar un archivo worker en webpack 5 / Next.js 13+.
    // El bundler lo empaqueta como un chunk separado con su propio scope.
    const worker = new Worker(
      new URL('../workers/onnx.worker.ts', import.meta.url),
      { type: 'module' } // Necesario para que el worker use imports ESM
    );
    workerRef.current = worker;

    // ── 3. Manejar mensajes del worker ────────────────────────────────────
    worker.addEventListener('message', (event: MessageEvent<WorkerOutMessage>) => {
      const msg = event.data;

      if (msg.type === 'READY') {
        setBackend(msg.backend);
        setStatus('');
        addLog(`✅ Listo | backend: ${msg.backend}`);
        // Iniciar el loop de render solo cuando el modelo está listo
        startRenderLoop(video, canvas, worker);
        return;
      }

      if (msg.type === 'RESULT') {
        // Actualizar las detecciones que el rAF dibujará en el próximo frame
        detections.current = msg.detections;
        setInferMs(msg.inferMs);
        setCount(msg.detections.length);
        // Liberar el semáforo para que el próximo frame se envíe
        inferBusy.current = false;
        return;
      }

      if (msg.type === 'LOG') {
        addLog(msg.text);
        return;
      }

      if (msg.type === 'ERROR') {
        console.error('[Worker error]', msg.message);
        setStatus(`Error: ${msg.message}`);
        addLog(`❌ ${msg.message}`);
        inferBusy.current = false;
      }
    });

    worker.addEventListener('error', (e) => {
      setStatus(`Worker crash: ${e.message}`);
      inferBusy.current = false;
    });

    // ── 4. Arrancar en paralelo ────────────────────────────────────────────
    startCamera().then(() => {
      // Enviamos el mensaje de init después de que la cámara esté lista,
      // pero la carga del modelo y la cámara son independientes.
      worker.postMessage({ type: 'INIT', modelUrl: MODEL_URL });
    });

    // ── Cleanup al desmontar el componente ─────────────────────────────────
    return () => {
      stopped = true;
      cancelAnimationFrame(rafId.current);
      stream?.getTracks().forEach(t => t.stop());
      worker.terminate();
    };
  }, [startRenderLoop]);

  // ─────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <div className="detector-root">
      <div className="viewport">
        {/* Video de la cámara */}
        <video
          ref={videoRef}
          className="camera-feed"
          autoPlay
          muted
          playsInline
          aria-label="Feed de cámara"
        />

        {/* Canvas superpuesto para bounding boxes */}
        <canvas
          ref={overlayRef}
          className="detection-overlay"
          aria-hidden="true"
        />

        {/* HUD de métricas */}
        <HUD
          backend={backend}
          inferMs={inferMs}
          fps={fps}
          count={count}
          status={status}
          modelMode={MODEL_MODE}
          logs={logs}
        />
      </div>
    </div>
  );
}
