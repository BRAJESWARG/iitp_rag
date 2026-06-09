// ─────────────────────────────────────────────────────────────
// hooks/useRAG.ts — Custom React hook for RAG pipeline state
// ─────────────────────────────────────────────────────────────

import { useState, useCallback, useEffect } from 'react';
import axios from 'axios';
import type {
  Message,
  PipelineState,
  StatusResponse,
  QueryResponse,
  IngestResponse,
} from '../types';

const API_BASE = '/api';

export function useRAG() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [pipeline, setPipeline] = useState<PipelineState>({ stage: 'idle' });
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);

  // ── Fetch system status ─────────────────────────────────────
  const fetchStatus = useCallback(async () => {
    try {
      const res = await axios.get<StatusResponse>(`${API_BASE}/status`);
      setStatus(res.data);
    } catch {
      setStatus(null);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10_000); // poll every 10s
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // ── Helper: Fallback to standard HTTP query ──────────────────
  const runHttpFallback = useCallback(async (query: string, assistantMsgId: string) => {
    // Animate through stages
    setPipeline({ stage: 'stage1_searching' });
    await new Promise((r) => setTimeout(r, 800));

    setPipeline({ stage: 'stage2_reranking' });
    await new Promise((r) => setTimeout(r, 600));

    setPipeline({ stage: 'llm_generating' });

    try {
      const res = await axios.post<QueryResponse>(`${API_BASE}/query`, {
        query: query.trim(),
      });

      const data = res.data;

      setPipeline({
        stage: 'done',
        stage1Count: data.stage1_candidates,
        stage2Count: data.stage2_top_docs,
      });

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? {
                ...m,
                content: data.answer,
                metadata: {
                  stage1_candidates: data.stage1_candidates,
                  stage2_top_docs: data.stage2_top_docs,
                  top_doc_scores: data.top_doc_scores,
                  top_doc_snippets: data.top_doc_snippets,
                  model_used: data.model_used,
                  duration_ms: data.duration_ms,
                },
              }
            : m
        )
      );
    } catch (err: unknown) {
      const errorMsg =
        axios.isAxiosError(err)
          ? err.response?.data?.detail || err.message
          : 'An unexpected error occurred.';

      setPipeline({ stage: 'error', errorMessage: errorMsg });

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? { ...m, content: `⚠️ Error: ${errorMsg}` }
            : m
        )
      );
    } finally {
      setIsLoading(false);
      setTimeout(() => setPipeline({ stage: 'idle' }), 2000);
    }
  }, []);

  // ── Send a query (WebSocket with HTTP fallback) ─────────────
  const sendQuery = useCallback(async (query: string) => {
    if (!query.trim() || isLoading) return;

    // Add user and empty assistant messages immediately
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
      timestamp: new Date(),
    };

    const assistantMsgId = crypto.randomUUID();
    const assistantMsg: Message = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsLoading(true);

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}${API_BASE}/ws/query`;

    let socket: WebSocket | null = null;
    let fallbackToHttp = false;

    // Helper to append metadata
    const updateAssistantMetadata = (meta: NonNullable<Message['metadata']>) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? { ...m, metadata: { ...m.metadata, ...meta } }
            : m
        )
      );
    };

    try {
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        socket?.send(JSON.stringify({ query: query.trim() }));
      };

      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'stage') {
          setPipeline({
            stage: msg.stage,
            stage1Count: msg.stage1_candidates,
            stage2Count: msg.stage2_top_docs,
          });

          if (msg.stage === 'llm_generating') {
            updateAssistantMetadata({
              stage1_candidates: msg.stage1_candidates,
              stage2_top_docs: msg.stage2_top_docs,
              top_doc_scores: msg.top_doc_scores,
              top_doc_snippets: msg.top_doc_snippets,
            });
          }
        } else if (msg.type === 'chunk') {
          // Append text chunk
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, content: m.content + msg.text }
                : m
            )
          );
        } else if (msg.type === 'done') {
          setPipeline({
            stage: 'done',
          });
          updateAssistantMetadata({
            duration_ms: msg.duration_ms,
            model_used: msg.model_used,
          });
          setIsLoading(false);
          socket?.close();
          setTimeout(() => setPipeline({ stage: 'idle' }), 2000);
        } else if (msg.type === 'error') {
          setPipeline({ stage: 'error', errorMessage: msg.message });
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, content: `⚠️ Error: ${msg.message}` }
                : m
            )
          );
          setIsLoading(false);
          socket?.close();
          setTimeout(() => setPipeline({ stage: 'idle' }), 2000);
        }
      };

      socket.onerror = (err) => {
        console.error('WebSocket Error:', err);
        fallbackToHttp = true;
        socket?.close();
      };

      socket.onclose = () => {
        if (fallbackToHttp) {
          runHttpFallback(query, assistantMsgId);
        }
      };

    } catch (err) {
      console.error('WebSocket Setup Error:', err);
      runHttpFallback(query, assistantMsgId);
    }
  }, [isLoading, runHttpFallback]);

  // ── Ingest documents ────────────────────────────────────────
  const ingestFiles = useCallback(async (files: File[]): Promise<IngestResponse> => {
    setIsIngesting(true);
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    try {
      const res = await axios.post<IngestResponse>(`${API_BASE}/ingest`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      await fetchStatus();
      return res.data;
    } finally {
      setIsIngesting(false);
    }
  }, [fetchStatus]);

  // ── Clear chat ──────────────────────────────────────────────
  const clearMessages = useCallback(() => {
    setMessages([]);
    setPipeline({ stage: 'idle' });
  }, []);

  return {
    messages,
    pipeline,
    status,
    isLoading,
    isIngesting,
    sendQuery,
    ingestFiles,
    clearMessages,
    fetchStatus,
  };
}
