'use client';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

const STORAGE_KEY = 'tractor:nickname';

export default function HomePage() {
  const router = useRouter();
  const [nick, setNick] = useState('');
  const [code, setCode] = useState('');

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved !== null) setNick(saved);
  }, []);

  function go(target: string) {
    if (nick.trim() === '') return;
    localStorage.setItem(STORAGE_KEY, nick.trim());
    router.push(`/room/${target}`);
  }

  function createRoom() {
    const newCode = Math.random().toString(36).slice(2, 6).toUpperCase();
    go(newCode);
  }

  function joinRoom() {
    const c = code.trim().toUpperCase();
    if (c === '') return;
    go(c);
  }

  return (
    <div className="center">
      <h1 className="h1">Tractor</h1>
      <p className="muted">Play 升级 / 拖拉机 in your browser.</p>

      <label>
        <div>Nickname</div>
        <input
          type="text"
          value={nick}
          onChange={(e) => setNick(e.target.value)}
          maxLength={32}
          placeholder="Your name"
        />
      </label>

      <button onClick={createRoom} disabled={nick.trim() === ''}>
        Create new room
      </button>

      <hr style={{ borderColor: '#2f4233', width: '100%' }} />

      <label>
        <div>Room code</div>
        <input
          type="text"
          value={code}
          onChange={(e) => setCode(e.target.value)}
          maxLength={8}
          placeholder="e.g. ABCD"
          style={{ textTransform: 'uppercase' }}
        />
      </label>

      <button onClick={joinRoom} disabled={nick.trim() === '' || code.trim() === ''}>
        Join room
      </button>
    </div>
  );
}
