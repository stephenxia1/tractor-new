# syntax=docker/dockerfile:1.7
# Image for the @tractor/ws WebSocket server. Builds from the monorepo
# root so npm can resolve the workspace deps (@tractor/engine, @tractor/protocol).
FROM node:20-alpine

WORKDIR /app

# Install workspace deps. Copying only manifests first keeps this layer
# cached across source-only changes.
COPY package.json package-lock.json tsconfig.base.json ./
COPY apps/ws/package.json apps/ws/package.json
COPY apps/web/package.json apps/web/package.json
COPY packages/engine/package.json packages/engine/package.json
COPY packages/protocol/package.json packages/protocol/package.json

# Install only the deps needed for the ws workspace + its transitive
# workspace packages. --include=dev because we run via `tsx`.
RUN npm ci --include=dev

# Bring in source for the workspaces the ws server actually uses.
COPY apps/ws apps/ws
COPY packages/engine packages/engine
COPY packages/protocol packages/protocol

ENV NODE_ENV=production
ENV PORT=8787
EXPOSE 8787

CMD ["npm", "run", "start", "--workspace", "@tractor/ws"]
