// ─────────────────────────────────────────────────────────────────
// components/Header.tsx — Top navigation bar
// ─────────────────────────────────────────────────────────────────
import React from 'react';
import { StatusResponse } from '../types';

interface HeaderProps {
  status: StatusResponse | null;
  onClear: () => void;
}

export const Header: React.FC<HeaderProps> = ({ status, onClear }) => {
  const isReady = status?.initialized;
  const isLoading = status?.status === 'initializing';

  return (
    <header className="app-header" style={{ justifyContent: 'space-between' }}>
      {/* Logo + Title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        <div style={{
          width: 38, height: 38, borderRadius: '10px',
          background: 'linear-gradient(135deg, #7c3aed, #06b6d4)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '18px', boxShadow: '0 0 20px rgba(124,58,237,0.4)',
          flexShrink: 0,
        }}>
          🔍
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontWeight: 700, fontSize: '16px' }}>
              <span className="gradient-text">Two-Stage RAG</span>
            </span>
            <span style={{
              fontSize: '11px', padding: '2px 8px', borderRadius: '20px',
              background: 'rgba(124,58,237,0.15)', color: '#a78bfa',
              border: '1px solid rgba(124,58,237,0.2)', fontWeight: 600,
            }}>
              v2.0
            </span>
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '1px' }}>
            BM25 + Vector → Cross Encoder → Gemini
          </div>
        </div>
      </div>

      {/* Pipeline Tags */}
      <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
        {[
          { label: 'Hybrid Search', color: '#3b82f6' },
          { label: 'Cross Encoder', color: '#8b5cf6' },
          { label: 'Gemini 2.5', color: '#f59e0b' },
        ].map((tag) => (
          <span key={tag.label} style={{
            fontSize: '11px', padding: '3px 10px', borderRadius: '20px',
            background: `${tag.color}15`, color: tag.color,
            border: `1px solid ${tag.color}25`,
          }}>
            {tag.label}
          </span>
        ))}
      </div>

      {/* Status + Actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {/* System Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {isLoading ? (
            <span style={{ fontSize: '13px', color: 'var(--warning)' }}>
              <span className="spin" style={{ display: 'inline-block' }}>⚙️</span> Initializing...
            </span>
          ) : isReady ? (
            <>
              <div className="pulse-dot" />
              <span style={{ fontSize: '13px', color: 'var(--success)' }}>
                Pipeline Ready
              </span>
            </>
          ) : (
            <>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--text-muted)' }} />
              <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                No documents
              </span>
            </>
          )}
        </div>

        <button className="btn btn-ghost" onClick={onClear} style={{ padding: '7px 14px', fontSize: '13px' }}>
          🗑 Clear Chat
        </button>
      </div>
    </header>
  );
};
