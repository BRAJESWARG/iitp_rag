// ─────────────────────────────────────────────────────────────────
// components/MessageBubble.tsx — Chat message with metadata
// ─────────────────────────────────────────────────────────────────
import React, { useState } from 'react';
import { Message } from '../types';

interface MessageBubbleProps {
  message: Message;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const [showSources, setShowSources] = useState(false);
  const isUser = message.role === 'user';
  const meta = message.metadata;

  const formattedTime = message.timestamp.toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit',
  });

  return (
    <div
      className="fade-in-up"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        gap: '6px',
        maxWidth: '85%',
        alignSelf: isUser ? 'flex-end' : 'flex-start',
      }}
    >
      {/* Avatar + Name */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '8px',
        flexDirection: isUser ? 'row-reverse' : 'row',
      }}>
        <div style={{
          width: 28, height: 28, borderRadius: '8px',
          background: isUser
            ? 'linear-gradient(135deg, #7c3aed, #4f46e5)'
            : 'linear-gradient(135deg, #0f172a, #1e293b)',
          border: isUser ? 'none' : '1px solid rgba(6,182,212,0.3)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '13px', flexShrink: 0,
          boxShadow: isUser ? '0 0 12px rgba(124,58,237,0.3)' : '0 0 12px rgba(6,182,212,0.15)',
        }}>
          {isUser ? '👤' : '🤖'}
        </div>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: 500 }}>
          {isUser ? 'You' : 'RAG Assistant'} · {formattedTime}
        </span>
        {!isUser && meta?.duration_ms && (
          <span style={{
            fontSize: '10px', padding: '2px 7px', borderRadius: '20px',
            background: 'rgba(16,185,129,0.1)', color: 'var(--success)',
            border: '1px solid rgba(16,185,129,0.15)',
          }}>
            {(meta.duration_ms / 1000).toFixed(2)}s
          </span>
        )}
      </div>

      {/* Bubble */}
      <div style={{
        padding: '14px 18px',
        borderRadius: isUser ? '16px 4px 16px 16px' : '4px 16px 16px 16px',
        background: isUser
          ? 'linear-gradient(135deg, rgba(124,58,237,0.25), rgba(79,70,229,0.2))'
          : 'rgba(255,255,255,0.04)',
        border: isUser
          ? '1px solid rgba(124,58,237,0.3)'
          : '1px solid rgba(255,255,255,0.07)',
        backdropFilter: 'blur(10px)',
        fontSize: '14px',
        lineHeight: '1.65',
        color: 'var(--text-primary)',
        maxWidth: '100%',
        wordBreak: 'break-word',
        whiteSpace: 'pre-wrap',
        boxShadow: isUser
          ? '0 4px 20px rgba(124,58,237,0.15)'
          : '0 4px 20px rgba(0,0,0,0.2)',
      }}>
        {message.content}
      </div>

      {/* Metadata bar for AI messages */}
      {!isUser && meta && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: '6px', alignItems: 'center',
          paddingLeft: '4px',
        }}>
          {meta.stage1_candidates !== undefined && (
            <span className="badge badge-info">
              🔍 {meta.stage1_candidates} candidates
            </span>
          )}
          {meta.stage2_top_docs !== undefined && (
            <span className="badge badge-purple">
              ⚡ Top {meta.stage2_top_docs} reranked
            </span>
          )}
          {meta.model_used && (
            <span className="badge badge-warning">
              ✨ {meta.model_used}
            </span>
          )}
          {meta.top_doc_snippets && meta.top_doc_snippets.length > 0 && (
            <button
              onClick={() => setShowSources(!showSources)}
              style={{
                fontSize: '11px', padding: '3px 10px', borderRadius: '20px',
                background: 'rgba(255,255,255,0.06)', color: 'var(--text-secondary)',
                border: '1px solid rgba(255,255,255,0.1)', cursor: 'pointer',
                fontFamily: 'inherit',
              }}
            >
              {showSources ? '▲ Hide sources' : '▼ View sources'}
            </button>
          )}
        </div>
      )}

      {/* Source documents */}
      {!isUser && showSources && meta?.top_doc_snippets && (
        <div
          className="fade-in-up"
          style={{
            width: '100%', display: 'flex', flexDirection: 'column', gap: '8px',
            paddingLeft: '4px',
          }}
        >
          {meta.top_doc_snippets.map((snippet, idx) => (
            <div
              key={idx}
              style={{
                padding: '10px 14px', borderRadius: '10px',
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.06)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', alignItems: 'center' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600 }}>
                  Source {idx + 1}
                </span>
                {meta.top_doc_scores?.[idx] !== undefined && (
                  <span style={{
                    fontSize: '10px', padding: '2px 8px', borderRadius: '20px',
                    background: 'rgba(139,92,246,0.12)', color: '#a78bfa',
                    border: '1px solid rgba(139,92,246,0.2)',
                    fontFamily: 'JetBrains Mono, monospace',
                  }}>
                    score: {meta.top_doc_scores[idx].toFixed(3)}
                  </span>
                )}
              </div>
              <p style={{
                fontSize: '12px', color: 'var(--text-secondary)',
                lineHeight: '1.55', margin: 0, fontFamily: 'JetBrains Mono, monospace',
              }}>
                {snippet}
                {snippet.length >= 200 ? '...' : ''}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
