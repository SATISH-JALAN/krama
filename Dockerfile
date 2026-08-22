# KRAMA server image. Currently built and run LOCALLY (no cloud account
# resolved yet -- see docs/MEMORY.md's deploy-blocker discussion) so this
# can be tested end-to-end today and swapped onto a real free host (Oracle/
# AWS/whichever) later without changing the app itself.
#
# hnswlib-node is an optionalDependency (package.json) -- its native
# postinstall (node-gyp rebuild) is allowed to fail here (no build-essential
# installed, deliberately, since HNSW is off the critical path per the
# approved brute-force-retrieval plan, see MEMORY.md). bruteforce.ts is the
# only dense-retrieval path this image actually needs.
FROM oven/bun:1

WORKDIR /app

ENV NODE_ENV=production \
    PORT=3000

COPY package.json bun.lock ./
RUN bun install --production

COPY server/ ./server/
COPY artifacts/onnx ./artifacts/onnx
COPY artifacts/onnx_reranker ./artifacts/onnx_reranker
COPY artifacts/thresholds.json ./artifacts/thresholds.json
COPY data/medium/passages_dedup.jsonl data/medium/embeddings.f32bin data/medium/embeddings_ids.json ./data/medium/

# HF Spaces/most PaaS platforms run containers as a non-root user by
# convention/requirement -- matching that here even for local runs so the
# image doesn't need changes when it's actually deployed. oven/bun's base
# image already ships a uid-1000 `bun` user, so reuse it instead of
# colliding with it.
RUN chown -R bun:bun /app
USER bun

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD bun -e "fetch('http://localhost:3000/health').then(r => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"

CMD ["bun", "run", "server/index.ts"]
