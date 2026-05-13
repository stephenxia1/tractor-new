// Quick end-to-end smoke test: opens 4 connections, joins a room, takes
// seats, starts a hand, and exits cleanly. Run with the server already
// listening on :8787.
import WebSocket from 'ws';
import type { ServerMessage } from '@tractor/protocol';

const PORT = Number(process.env.PORT ?? 8787);
const URL = `ws://localhost:${PORT}/ws`;

function delay(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

type Conn = {
  ws: WebSocket;
  clientId: string | null;
  lastRoomState: ServerMessage | null;
};

function connect(nickname: string, roomCode: string): Promise<Conn> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(URL);
    const conn: Conn = { ws, clientId: null, lastRoomState: null };
    ws.on('open', () => {
      ws.send(JSON.stringify({ kind: 'join', nickname, roomCode }));
    });
    ws.on('message', (raw) => {
      const msg = JSON.parse(String(raw)) as ServerMessage;
      if (msg.kind === 'welcome') conn.clientId = msg.clientId;
      if (msg.kind === 'room_state') conn.lastRoomState = msg;
      if (conn.clientId !== null && conn.lastRoomState !== null) resolve(conn);
    });
    ws.on('error', reject);
  });
}

async function main() {
  console.log('connecting 4 players...');
  const a = await connect('alice', 'TEST');
  const b = await connect('bob', 'TEST');
  const c = await connect('carol', 'TEST');
  const d = await connect('dave', 'TEST');

  a.ws.send(JSON.stringify({ kind: 'take_seat', seat: 0 }));
  b.ws.send(JSON.stringify({ kind: 'take_seat', seat: 1 }));
  c.ws.send(JSON.stringify({ kind: 'take_seat', seat: 2 }));
  d.ws.send(JSON.stringify({ kind: 'take_seat', seat: 3 }));
  await delay(200);

  console.log('starting hand...');
  a.ws.send(JSON.stringify({ kind: 'start_hand' }));
  await delay(200);

  console.log('hand started; closing.');
  for (const x of [a, b, c, d]) x.ws.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
