# Grape Detection PoC

Real-time grape berry detection in mobile browsers using ONNX Runtime Web and Next.js.

- **Next.js 15** (App Router, TypeScript)
- **ONNX Runtime Web 1.21** (WebGPU → WebGL → WASM fallback)
- **YOLOv8n** fine-tuned for grape cluster detection
- **Web Worker** for off-main-thread inference
- **60 FPS** UI with async inference via "one-at-a-time" backpressure

## Models

| Mode | Model | Classes |
|------|-------|---------|
| `coco` | YOLOv8n (default) | 80 COCO classes |
| `grape` | YOLOv8n fine-tuned | Grape clusters |

The grape model was trained on [grape-detection-xcemr](https://universe.roboflow.com/grape-detection/grape-detection-xcemr) via Google Colab (10 epochs, T4 GPU). Training notebook: `scripts/train_grape_colab.ipynb`.

## Environment variables

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_MODEL_URL` | ONNX model URL (Vercel Blob or `/models/filename.onnx`) |
| `NEXT_PUBLIC_MODEL_MODE` | `coco` or `grape` |

## File structure

```
src/
├── app/
│   ├── layout.tsx          # Root layout (Server Component)
│   ├── page.tsx            # Página principal (Server Component)
│   └── globals.css         # Estilos globales mínimos
├── components/
│   ├── Detector.tsx        # Lógica de cámara + rAF + canvas (Client Component)
│   └── HUD.tsx             # Métricas en tiempo real
├── workers/
│   └── onnx.worker.ts      # Worker: ORT init + inferencia + NMS
└── lib/
    ├── constants.ts        # Tipos, etiquetas COCO, colores
    └── preprocessing.ts    # bitmapToTensor + decodeAndNMS
public/
└── models/
    └── yolov8n.onnx        # ← DEBES COLOCAR EL MODELO AQUÍ
```

## 1. Obtener el modelo ONNX

```bash
# Opción A: con Python + ultralytics
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', opset=17, simplify=True)"
cp yolov8n.onnx public/models/

# Opción B: descarga directa (modelo pre-exportado de la comunidad)
# https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx
curl -L -o public/models/yolov8n.onnx \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx
```

## 2. Instalar dependencias y arrancar

```bash
npm install
npm run dev
```

Abre `http://localhost:3000` en Chrome para Android (o Chrome desktop).
Para iOS Safari necesitas habilitar WebGPU en Ajustes → Safari → Características experimentales.

## 3. Decisiones de arquitectura clave

| Decisión | Por qué |
|---|---|
| Web Worker dedicado | La inferencia puede tardar 20–150 ms; en el main thread congelaría la UI |
| ImageBitmap transferible | Zero-copy entre main thread y worker; evita duplicar ~5 MB por frame |
| `willReadFrequently` en OffscreenCanvas | Mantiene copia CPU-accesible; evita readbacks costosos desde GPU |
| `inferBusy` flag (backpressure) | El rAF va a 60 FPS pero solo envía un frame a la vez al worker |
| `createImageBitmap` con resize | Delega el escalado al compositor (GPU) en lugar de hacerlo en CPU |
| `graphOptimizationLevel: 'all'` | Aumenta tiempo de carga ~15% pero reduce latencia por inferencia |
| `freeDimensionOverrides` | Permite al compilador de operaciones generar código para tamaño fijo |

## 4. Métricas esperadas (referencia)

| Dispositivo | Backend | Inferencia | UI FPS |
|---|---|---|---|
| Pixel 8 (Chrome) | WebGPU | ~15–25 ms | 60 |
| Samsung S22 (Chrome) | WebGPU | ~20–35 ms | 60 |
| iPhone 15 (Safari) | WebGL | ~30–60 ms | 60 |
| Laptop M2 (Chrome) | WebGPU | ~5–10 ms | 60 |
| Dispositivo bajo (WASM) | WASM | ~80–200 ms | 60* |

*La UI siempre va a 60 FPS; la inferencia es asíncrona en el worker.
