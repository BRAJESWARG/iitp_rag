# Two-Stage RAG — Node proxy: builds the React UI and serves it + proxies /api.
# Build context = the two_stage_rag/ directory (so it can see ../frontend).

# ---- Stage 1: build the React frontend (Vite 8 needs Node 20+) ----
FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ---- Stage 2: the Node/Express proxy ----
FROM node:20-slim AS proxy
WORKDIR /app/backend
COPY backend/package.json backend/package-lock.json ./
RUN npm ci
COPY backend/ .
# server.ts serves ../../frontend/dist relative to its own dir → /app/frontend/dist
COPY --from=frontend /frontend/dist /app/frontend/dist

EXPOSE 3001
# Run via tsx (matches the working dev path; no separate tsc step needed)
CMD ["npx", "tsx", "src/server.ts"]
