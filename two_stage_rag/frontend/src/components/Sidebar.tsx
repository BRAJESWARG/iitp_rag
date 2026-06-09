// ─────────────────────────────────────────────────────────────────
// components/Sidebar.tsx — Left sidebar with system info
// ─────────────────────────────────────────────────────────────────
import React from 'react';
import { StatusResponse } from '../types';
import { DocumentUpload } from './DocumentUpload';
import { PipelineViz } from './PipelineViz';
import { PipelineState } from '../types';
import { IngestResponse } from '../types';

interface SidebarProps {
  status: StatusResponse | null;
  pipeline: PipelineState;
  onIngest: (files: File[]) => Promise<IngestResponse>;
  isIngesting: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  status, pipeline, onIngest, isIngesting,
}) => {
  return (
    <aside className="app-sidebar">
      {/* Document Upload */}
      <DocumentUpload onIngest={onIngest} isIngesting={isIngesting} />

      {/* Pipeline Visualization */}
      <PipelineViz pipeline={pipeline} />

      {/* System Info */}
      <div className="glass-card" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          System Info
        </span>

        {status ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {/* Models */}
            {[
              { label: 'Bi-Encoder', value: 'all-MiniLM-L6-v2', color: '#3b82f6' },
              { label: 'Reranker', value: 'ms-marco-MiniLM-L-6-v2', color: '#8b5cf6' },
              { label: 'LLM', value: status.model_llm || 'gemini-2.5-flash', color: '#f59e0b' },
            ].map(({ label, value, color }) => (
              <div key={label}>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '3px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  {label}
                </div>
                <div style={{
                  fontSize: '11px', fontFamily: 'JetBrains Mono, monospace',
                  color, padding: '4px 8px', borderRadius: '6px',
                  background: `${color}10`, border: `1px solid ${color}20`,
                  wordBreak: 'break-all',
                }}>
                  {value}
                </div>
              </div>
            ))}

            {/* Stats */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '4px' }}>
              <div style={{
                padding: '10px', borderRadius: '8px', textAlign: 'center',
                background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.15)',
              }}>
                <div style={{ fontSize: '20px', fontWeight: 700, color: '#3b82f6' }}>
                  {status.chunks_count || 0}
                </div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>Chunks</div>
              </div>
              <div style={{
                padding: '10px', borderRadius: '8px', textAlign: 'center',
                background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.15)',
              }}>
                <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--success)' }}>
                  {status.documents_loaded?.length || 0}
                </div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>Docs</div>
              </div>
            </div>

            {/* Loaded files */}
            {status.documents_loaded?.length > 0 && (
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '5px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  Loaded Files
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  {status.documents_loaded.map((doc) => (
                    <div key={doc} style={{
                      fontSize: '11px', padding: '5px 8px', borderRadius: '6px',
                      background: 'rgba(255,255,255,0.04)',
                      color: 'var(--text-secondary)',
                      display: 'flex', alignItems: 'center', gap: '6px',
                    }}>
                      <span>📄</span>
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {doc}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* API Key status */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              padding: '8px 10px', borderRadius: '8px',
              background: status.api_key_set ? 'rgba(16,185,129,0.07)' : 'rgba(239,68,68,0.07)',
              border: `1px solid ${status.api_key_set ? 'rgba(16,185,129,0.2)' : 'rgba(239,68,68,0.2)'}`,
            }}>
              <span style={{ fontSize: '14px' }}>{status.api_key_set ? '🔑' : '⚠️'}</span>
              <span style={{
                fontSize: '11px',
                color: status.api_key_set ? 'var(--success)' : 'var(--error)',
              }}>
                {status.api_key_set ? 'Gemini API key set' : 'API key missing'}
              </span>
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {[1, 2, 3].map((i) => (
              <div key={i} className="shimmer" style={{ height: '36px', borderRadius: '8px' }} />
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{ marginTop: 'auto', padding: '8px 4px', fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center' }}>
        Two-Stage RAG Pipeline · IITP Workshop
      </div>
    </aside>
  );
};
