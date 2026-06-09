/**
 * server.ts — Node.js Express Backend
 * =====================================
 * Acts as a proxy between the React frontend and the Python FastAPI server.
 * Also serves the built React production bundle.
 *
 * Endpoints (proxied to Python FastAPI on port 8000):
 *   GET  /api/status   → System status
 *   POST /api/query    → Run RAG pipeline
 *   POST /api/ingest   → Upload and ingest documents
 */

import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import { createProxyMiddleware } from 'http-proxy-middleware';
import path from 'path';
import dotenv from 'dotenv';

dotenv.config({ path: path.join(__dirname, '../../.env') });

const app = express();
const PORT = process.env.NODE_PORT || 3001;
const PYTHON_API = process.env.PYTHON_API_URL || 'http://localhost:8000';

// ──────────────────────────────────────────────────────
// Middleware
// ──────────────────────────────────────────────────────
app.use(cors({
  origin: ['http://localhost:5173', 'http://localhost:3000', 'http://localhost:3001'],
  credentials: true,
}));

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// ──────────────────────────────────────────────────────
// Health check
// ──────────────────────────────────────────────────────
app.get('/health', (_req: Request, res: Response) => {
  res.json({
    status: 'ok',
    node_server: 'running',
    python_api: PYTHON_API,
    timestamp: new Date().toISOString(),
  });
});

// ──────────────────────────────────────────────────────
// Proxy all /api/* calls to Python FastAPI (including WebSockets)
// ──────────────────────────────────────────────────────
const apiProxy = createProxyMiddleware({
  target: PYTHON_API,
  changeOrigin: true,
  ws: true, // Enable WebSocket proxying
  onError: (err: Error, _req: any, res: any) => {
    console.error('[Proxy Error]', err.message);
    if (res.status) {
      res.status(502).json({
        error: 'Python API unavailable',
        detail: 'Make sure the Python FastAPI server is running on port 8000.\n' +
                'Run: cd two_stage_rag && source venv/bin/activate && uvicorn api:app --port 8000',
      });
    }
  },
});

app.use('/api', apiProxy);

// ──────────────────────────────────────────────────────
// Serve React production build (if it exists)
// ──────────────────────────────────────────────────────
const frontendDist = path.join(__dirname, '../../frontend/dist');
app.use(express.static(frontendDist));

// SPA fallback — return index.html for all non-API routes
app.get('*', (_req: Request, res: Response) => {
  const indexPath = path.join(frontendDist, 'index.html');
  res.sendFile(indexPath, (err) => {
    if (err) {
      res.status(200).json({
        message: 'Two-Stage RAG API running',
        hint: 'Build the React frontend with: cd frontend && npm run build',
        api_docs: `http://localhost:8000/docs`,
      });
    }
  });
});

// ──────────────────────────────────────────────────────
// Error handler
// ──────────────────────────────────────────────────────
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  console.error('[Server Error]', err);
  res.status(500).json({ error: err.message });
});

// ──────────────────────────────────────────────────────
// Start
// ──────────────────────────────────────────────────────
const server = app.listen(PORT, () => {
  console.log('\n' + '═'.repeat(55));
  console.log(' Two-Stage RAG — Node.js Backend');
  console.log('═'.repeat(55));
  console.log(`  Node.js server : http://localhost:${PORT}`);
  console.log(`  Python API     : ${PYTHON_API}`);
  console.log(`  Python API docs: ${PYTHON_API}/docs`);
  console.log('═'.repeat(55) + '\n');
});

// Upgrade WebSocket connections proxied to FastAPI
server.on('upgrade', (request, socket, head) => {
  if (request.url?.startsWith('/api')) {
    (apiProxy as any).upgrade(request, socket, head);
  }
});
