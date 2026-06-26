/**
 * scripts/upload-model.ts
 * Sube un modelo ONNX a Vercel Blob y muestra la URL pública.
 *
 * Uso:
 *   npm run upload-model                        → sube yolov8n.onnx
 *   npm run upload-model -- grape-berry.onnx    → sube grape-berry.onnx
 *
 * Requiere:
 *   BLOB_READ_WRITE_TOKEN en el entorno (obtenido de Vercel Dashboard → Storage → Blob)
 */
import { put } from '@vercel/blob';
import { readFileSync } from 'fs';
import { join } from 'path';

// El nombre de archivo puede venir como argumento CLI (ej: "grape-berry.onnx")
const filename  = process.argv[2] ?? 'yolov8n.onnx';
const MODEL_PATH = join(process.cwd(), 'public', 'models', filename);
const MODEL_NAME = `models/${filename}`;

async function main() {
  if (!process.env.BLOB_READ_WRITE_TOKEN) {
    console.error('❌  Falta BLOB_READ_WRITE_TOKEN en el entorno.');
    console.error('   Obtén el token en: Vercel Dashboard → tu proyecto → Storage → Blob → Connect');
    process.exit(1);
  }

  console.log('📦  Leyendo modelo desde:', MODEL_PATH);
  const file = readFileSync(MODEL_PATH);

  console.log('⬆️   Subiendo a Vercel Blob...');
  const blob = await put(MODEL_NAME, file, {
    access: 'public',
    contentType: 'application/octet-stream',
    addRandomSuffix: false,
  });

  console.log('\n✅  Modelo subido correctamente!');
  console.log('🔗  URL pública:', blob.url);
  console.log('\n👉  Agrega esta variable de entorno en Vercel:');
  console.log(`   NEXT_PUBLIC_MODEL_URL=${blob.url}`);

  if (filename.includes('grape')) {
    console.log('   NEXT_PUBLIC_MODEL_MODE=grape');
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
