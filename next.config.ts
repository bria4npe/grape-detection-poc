import type { NextConfig } from 'next';
import { copyFileSync, mkdirSync, readdirSync, existsSync } from 'fs';
import path from 'path';

// ── Copiar archivos WASM de onnxruntime-web a /public/ort/ ───────────────────
// Webpack no puede bundlear los .wasm de ORT (genera el error "Can't resolve 'a'").
// La solución es servirlos como assets estáticos desde /public/ort/ y decirle
// a ORT en tiempo de ejecución dónde encontrarlos vía ort.env.wasm.wasmPaths.
// Este bloque corre en tiempo de build/dev, no en el browser.
try {
  const ortDist = path.resolve('./node_modules/onnxruntime-web/dist');
  const dest    = path.resolve('./public/ort');
  mkdirSync(dest, { recursive: true });
  readdirSync(ortDist)
    .filter(f => f.endsWith('.wasm') || f.endsWith('.mjs'))
    .forEach(f => {
      const target = path.join(dest, f);
      if (!existsSync(target)) copyFileSync(path.join(ortDist, f), target);
    });
  console.log('[next.config] ORT WASM files copied to /public/ort/');
} catch (e) {
  console.warn('[next.config] Could not copy ORT WASM files:', e);
}

const nextConfig: NextConfig = {
  // ── Headers HTTP para habilitar SharedArrayBuffer ────────────────────────
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'Cross-Origin-Opener-Policy',   value: 'same-origin'    },
          { key: 'Cross-Origin-Embedder-Policy',  value: 'require-corp'   },
          { key: 'Cross-Origin-Resource-Policy',  value: 'cross-origin'   },
        ],
      },
    ];
  },

  webpack(config, { isServer }) {
    if (!isServer) {
      // ── Excluir .wasm del pipeline de webpack ──────────────────────────
      // Usamos 'asset/resource' para que webpack los emita como archivos
      // separados en lugar de intentar parsearlos como módulos JS.
      // ORT los cargará directamente via fetch() usando la ruta /ort/*.
      config.module.rules.push({
        test: /\.wasm$/,
        type: 'asset/resource',
        generator: { filename: 'static/wasm/[name][ext]' },
      });

      // ── Workers con new URL() ─────────────────────────────────────────
      config.output = {
        ...config.output,
        workerChunkLoading: 'import-scripts',
      };
    }

    return config;
  },
};

export default nextConfig;
