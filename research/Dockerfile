# Training image for the RL research stack: Node (the game workers) and
# Python/torch (the learner) in one container, because the two halves talk
# over pipes and have to live together.
#
# Build from the REPO ROOT, not from research/, so npm can resolve the
# workspace link to @tractor/engine:
#
#   docker build -f research/Dockerfile -t tractor-train .
#   docker run --rm -v $PWD/out:/app/research/checkpoints tractor-train \
#       --workers 12 --games 64 --threads 8 --groups 2 --minutes 600
#
# For a CUDA box, override the wheel index at build time:
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121

FROM node:20-bookworm-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Workspace manifests first: this layer caches across source-only edits.
# Every workspace member's package.json must be present for `npm ci`.
COPY package.json package-lock.json tsconfig.base.json ./
COPY apps/ws/package.json apps/ws/package.json
COPY apps/web/package.json apps/web/package.json
COPY packages/engine/package.json packages/engine/package.json
COPY packages/protocol/package.json packages/protocol/package.json
RUN npm ci --include=dev

# torch is the big layer; install it before copying source so source edits
# do not re-download ~200MB.
RUN pip3 install --no-cache-dir --break-system-packages \
    numpy "torch>=2.0" --index-url ${TORCH_INDEX} \
    || pip3 install --no-cache-dir --break-system-packages numpy "torch>=2.0"

# The rules engine and the research stack are the only sources needed.
COPY packages/engine packages/engine
COPY packages/protocol packages/protocol
COPY research research
COPY rules.md rules.md

WORKDIR /app/research
ENV PYTHONPATH=/app/research/py
ENV PYTHONUNBUFFERED=1

# Fail fast on a broken image rather than after an hour of training.
RUN npx tsx src/dupcheck.ts --hands 3 --games 1 \
    && HANDS=5 npx tsx src/selftest.ts \
    && python3 py/test_returns.py

ENTRYPOINT ["python3", "py/train_dmc.py"]
CMD ["--workers", "8", "--games", "64", "--threads", "8", "--groups", "2", "--minutes", "600"]
