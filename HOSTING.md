# Hosting

Two-piece deployment:

- **Web (Next.js)** → Vercel
- **WebSocket server** → Fly.io

The web client talks to the WS server via `wss://...`, so both pieces
must be reachable over HTTPS and the WS server must accept the web
app's Origin.

---

## 1. Deploy the WS server to Fly.io

### 1.1 One-time setup

```bash
# Install the Fly CLI: https://fly.io/docs/hands-on/install-flyctl/
fly auth login
```

Pick a globally-unique app name and a region close to you
(`fly platform regions`). Edit `fly.toml` at the repo root:

```toml
app = "tractor-ws-yourname"   # must be globally unique
primary_region = "iad"        # or sjc, ord, lhr, etc.
```

### 1.2 Create the app and deploy

From the repo root (where `fly.toml` and `Dockerfile` live):

```bash
fly apps create tractor-ws-yourname
fly deploy
```

The first deploy may take a few minutes (Docker image build + image
push). Subsequent deploys are faster thanks to layer caching.

Verify it's up:

```bash
curl https://tractor-ws-yourname.fly.dev/health   # → ok
```

### 1.3 Lock the origin (after you know your Vercel URL)

```bash
fly secrets set ALLOWED_ORIGINS="https://your-app.vercel.app"
```

You can include multiple comma-separated origins (e.g. a preview alias
plus production). If `ALLOWED_ORIGINS` is unset, all origins are
allowed — convenient for local testing, but tighten it before you
share the link.

### 1.4 Useful Fly commands

```bash
fly logs                       # tail server logs
fly status                     # machine status
fly ssh console                # shell into the running container
fly scale memory 512           # bump memory if needed
fly machine list               # see all machines
```

---

## 2. Deploy the web app to Vercel

### 2.1 Push the repo to GitHub

Vercel auto-deploys from a Git repo. Push to GitHub (or GitLab /
Bitbucket) if you haven't already.

### 2.2 Import the project in Vercel

1. https://vercel.com/new → "Import" the GitHub repo.
2. **Root Directory**: leave as the repo root (the `vercel.json` at
   the root tells Vercel to build the `@tractor/web` workspace).
3. **Framework Preset**: Next.js (auto-detected via `vercel.json`).
4. **Build & Output settings**: leave defaults — `vercel.json`
   overrides them:
   - install: `npm ci`
   - build: `npm run build --workspace @tractor/web`
   - output: `apps/web/.next`

### 2.3 Set the WS URL env var

In the Vercel project → Settings → Environment Variables, add:

| Name                  | Value                                          |
| --------------------- | ---------------------------------------------- |
| `NEXT_PUBLIC_WS_URL`  | `wss://tractor-ws-yourname.fly.dev/ws`         |

Apply it to **Production**, **Preview**, and **Development** as
appropriate. Redeploy (Deployments → ⋯ → Redeploy) so the new env var
is baked into the client bundle.

### 2.4 Confirm

Open the Vercel URL, create a room, open browser devtools → Network →
WS, and confirm a connection to `wss://tractor-ws-yourname.fly.dev/ws`.

---

## 3. Environment variables — summary

| Service | Variable             | Where to set                       | Notes                                            |
| ------- | -------------------- | ---------------------------------- | ------------------------------------------------ |
| Web     | `NEXT_PUBLIC_WS_URL` | Vercel Project → Env Variables     | Must be `wss://`, must include `/ws` path        |
| WS      | `PORT`               | `fly.toml` (already 8787)          | Fly maps :80/:443 → internal_port for you        |
| WS      | `ALLOWED_ORIGINS`    | `fly secrets set ALLOWED_ORIGINS`  | Comma-separated list of allowed browser origins  |

---

## 4. Local development

Unchanged — the env-var indirection is opt-in:

```bash
# terminal 1
npm run dev --workspace @tractor/ws        # ws://localhost:8787/ws

# terminal 2
npm run dev --workspace @tractor/web       # http://localhost:3000
```

The web client falls back to `ws://localhost:8787/ws` when
`NEXT_PUBLIC_WS_URL` is unset.

---

## 5. Operational notes

- **State is in memory.** Rooms, sessions, and active games live in the
  WS server process. A Fly redeploy or crash drops every active room.
  This is fine for a small project; if you want persistence, the
  natural step is to add Redis (Fly has a managed addon).
- **Single-region by default.** Fly can scale to multiple regions, but
  with in-memory state you'd want sticky routing or shared storage
  first. Start single-region.
- **Cost.** A 256 MB shared-cpu Fly machine is ~$2/mo and easily fits
  a handful of concurrent tables. Vercel's hobby tier covers the
  Next.js app for free.
- **Scaling up.** `fly scale count 2` runs two machines, but you'd
  need to either pin clients to a machine or add a shared store
  before the rooms-in-memory model is safe across machines.
