/**
 * page.tsx — Server Component raíz.
 *
 * Este archivo corre en el servidor (o se pre-renderiza en build).
 * No puede usar hooks ni APIs del browser.
 * Solo compone el Detector, que es un Client Component.
 */
import { Detector } from '@/components/Detector';

export default function Home() {
  return (
    <main>
      <Detector />
    </main>
  );
}
