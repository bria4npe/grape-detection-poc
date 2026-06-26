/**
 * HUD.tsx
 * Overlay de métricas en tiempo real. Componente puro sin estado propio;
 * recibe todos los valores por props para que el padre controle el ciclo de vida.
 */
'use client';

interface HUDProps {
  backend:    string;
  inferMs:    number | null;
  fps:        number | null;
  count:      number | null;
  status:     string;
  modelMode?: 'coco' | 'grape';
  logs?:      string[];
}

export function HUD({ backend, inferMs, fps, count, status, modelMode = 'coco', logs = [] }: HUDProps) {
  const isGrape = modelMode === 'grape';
  return (
    <div className="hud">
      <div>Backend: <strong>{backend || '—'}</strong></div>
      <div>Inferencia: <strong>{inferMs != null ? `${inferMs.toFixed(1)} ms` : '—'}</strong></div>
      <div>FPS render: <strong>{fps != null ? fps.toFixed(0) : '—'}</strong></div>
      {isGrape
        ? <div>🍇 Bayas: <strong>{count ?? '—'}</strong></div>
        : <div>Detecciones: <strong>{count ?? '—'}</strong></div>
      }
      {status && <div className="hud-status">{status}</div>}
      {logs.length > 0 && (
        <div style={{ marginTop: 6, borderTop: '1px solid rgba(255,255,255,0.3)', paddingTop: 4 }}>
          {logs.map((l, i) => (
            <div key={i} style={{ fontSize: 10, opacity: 0.85, fontFamily: 'monospace', lineHeight: 1.3 }}>{l}</div>
          ))}
        </div>
      )}
    </div>
  );
}
