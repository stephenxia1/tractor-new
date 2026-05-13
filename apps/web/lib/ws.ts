'use client';
import { useEffect, useRef, useState, useCallback } from 'react';
import type { Card } from '@tractor/engine';
import type {
  ClientMessage,
  PrivateState,
  RoomView,
  ServerMessage,
} from '@tractor/protocol';

export type WsState = {
  readonly clientId: string | null;
  readonly room: RoomView | null;
  readonly priv: PrivateState | null;
  // Every seat's cards, populated when spectate mode is on. Null otherwise.
  readonly spectateHands: Readonly<Record<number, ReadonlyArray<Card>>> | null;
  readonly errors: ReadonlyArray<string>;
  readonly connected: boolean;
};

const WS_URL =
  (typeof window !== 'undefined' &&
    (window as unknown as { __WS_URL__?: string }).__WS_URL__) ||
  process.env.NEXT_PUBLIC_WS_URL ||
  'ws://localhost:8787/ws';

const SESSION_KEY = 'tractor:session';

export function getOrCreateSessionToken(): string {
  if (typeof window === 'undefined') return '';
  let token = localStorage.getItem(SESSION_KEY);
  if (token === null || token === '') {
    token = Math.random().toString(36).slice(2) + Math.random().toString(36).slice(2);
    localStorage.setItem(SESSION_KEY, token);
  }
  return token;
}

export function useTractorSocket(): {
  state: WsState;
  send: (msg: ClientMessage) => void;
} {
  const [state, setState] = useState<WsState>({
    clientId: null,
    room: null,
    priv: null,
    spectateHands: null,
    errors: [],
    connected: false,
  });
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    let backoff = 500;

    function connect() {
      if (cancelled) return;
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => {
        backoff = 500;
        setState((s) => ({ ...s, connected: true }));
      };
      ws.onmessage = (ev) => {
        const msg = JSON.parse(String(ev.data)) as ServerMessage;
        setState((s) => applyServerMessage(s, msg));
      };
      ws.onclose = () => {
        setState((s) => ({ ...s, connected: false }));
        if (cancelled) return;
        setTimeout(connect, backoff);
        backoff = Math.min(backoff * 2, 8000);
      };
      ws.onerror = () => {
        ws.close();
      };
    }

    connect();
    return () => {
      cancelled = true;
      wsRef.current?.close();
    };
  }, []);

  const send = useCallback((msg: ClientMessage) => {
    const ws = wsRef.current;
    if (ws === null || ws.readyState !== ws.OPEN) return;
    ws.send(JSON.stringify(msg));
  }, []);

  return { state, send };
}

function applyServerMessage(s: WsState, msg: ServerMessage): WsState {
  switch (msg.kind) {
    case 'welcome':
      return { ...s, clientId: msg.clientId };
    case 'room_state':
      return { ...s, room: msg.room };
    case 'private_state':
      return { ...s, priv: msg.state };
    case 'spectate_state':
      return { ...s, spectateHands: msg.hands };
    case 'error':
      return {
        ...s,
        errors: [...s.errors.slice(-9), msg.error],
      };
    case 'chat':
      return s; // chat not implemented in UI yet
  }
}
