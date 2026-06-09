// ─────────────────────────────────────────────────────────────
// types/index.ts — Shared TypeScript types for the RAG frontend
// ─────────────────────────────────────────────────────────────

export interface QueryResponse {
  answer: string;
  query: string;
  stage1_candidates: number;
  stage2_top_docs: number;
  top_doc_scores: number[];
  top_doc_snippets: string[];
  model_used: string;
  duration_ms: number;
}

export interface StatusResponse {
  status: 'ready' | 'not_initialized' | 'initializing' | 'error';
  initialized: boolean;
  documents_loaded: string[];
  chunks_count: number;
  model_stage1_embed: string;
  model_stage2_reranker: string;
  model_llm: string;
  api_key_set: boolean;
  started_at: string | null;
  error: string | null;
}

export interface IngestResponse {
  success: boolean;
  message: string;
  files_ingested: string[];
  total_chunks: number;
}

export type PipelineStage =
  | 'idle'
  | 'stage1_searching'
  | 'stage2_reranking'
  | 'llm_generating'
  | 'done'
  | 'error';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    stage1_candidates?: number;
    stage2_top_docs?: number;
    top_doc_scores?: number[];
    top_doc_snippets?: string[];
    model_used?: string;
    duration_ms?: number;
  };
}

export interface PipelineState {
  stage: PipelineStage;
  stage1Count?: number;
  stage2Count?: number;
  errorMessage?: string;
}
