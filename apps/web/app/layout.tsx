import './globals.css';
import type { ReactNode } from 'react';

export const metadata = {
  title: 'Tractor',
  description: 'Play Tractor (升级/拖拉机) online.',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
