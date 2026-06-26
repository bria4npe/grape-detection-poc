import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'YOLO ONNX PoC',
  description: 'Detección de objetos en tiempo real con ONNX Runtime Web y WebGPU',
};

export const viewport: Viewport = {
  // Desactivar zoom táctil en móvil para una experiencia de app nativa
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  // Usar toda la altura del dispositivo incluyendo la barra de estado iOS
  viewportFit: 'cover',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
