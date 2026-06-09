// ─────────────────────────────────────────────────────────────────
// components/ChatWindow.tsx — Main chat area with input bar
// ─────────────────────────────────────────────────────────────────
import React, { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { Message, PipelineState } from '../types';
import { MessageBubble } from './MessageBubble';

interface ChatWindowProps {
  messages: Message[];
  pipeline: PipelineState;
  isLoading: boolean;
  isInitialized: boolean;
  onSend: (query: string) => void;
}

const EXAMPLE_QUERIES = [
  'What is a transformer model?',
  'How does BM25 retrieval work?',
  'Explain the cross encoder vs bi-encoder difference',
  'What is Retrieval-Augmented Generation?',
  'How does multi-head attention work?',
];

export const ChatWindow: React.FC<ChatWindowProps> = ({
  messages, pipeline, isLoading, isInitialized, onSend,
}) => {
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 140) + 'px';
  }, [input]);

  const handleSend = () => {
    if (!input.trim() || isLoading || !isInitialized) return;
    onSend(input.trim());
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app-main">
      {/* Messages area */}
      <div style={{
        flex: 1, overflowY: 'auto', padding: '24px',
        display: 'flex', flexDirection: 'column', gap: '20px',
      }}>
        {messages.length === 0 ? (
          /* Empty state */
          <div style={{
            flex: 1, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            gap: '32px', padding: '40px',
          }}>
            {/* Hero */}
            <div style={{ textAlign: 'center' }}>
              <div style={{
                width: 72, height: 72, borderRadius: '20px', margin: '0 auto 20px',
                background: 'linear-gradient(135deg, #7c3aed22, #06b6d422)',
                border: '1px solid rgba(124,58,237,0.2)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '32px', boxShadow: '0 0 40px rgba(124,58,237,0.1)',
              }}>
                🔍
              </div>
              <h1 style={{
                fontSize: '24px', fontWeight: 700, marginBottom: '10px',
                background: 'linear-gradient(135deg, #7c3aed, #06b6d4)',
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              }}>
                Two-Stage RAG Pipeline
              </h1>
              <p style={{ color: 'var(--text-secondary)', maxWidth: '480px', lineHeight: '1.7', fontSize: '14px' }}>
                Ask any question about your documents. The pipeline uses{' '}
                <strong style={{ color: '#3b82f6' }}>Hybrid Search</strong> (BM25 + Vector) for recall,{' '}
                <strong style={{ color: '#8b5cf6' }}>Cross Encoder</strong> for precision reranking, then{' '}
                <strong style={{ color: '#f59e0b' }}>Gemini</strong> to generate a grounded answer.
              </p>
            </div>

            {/* Status message */}
            {!isInitialized && (
              <div style={{
                padding: '14px 20px', borderRadius: '12px',
                background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)',
                fontSize: '13px', color: 'var(--warning)', maxWidth: '420px', textAlign: 'center',
              }}>
                📄 Upload documents using the sidebar to get started
              </div>
            )}

            {/* Example queries */}
            {isInitialized && (
              <div style={{ width: '100%', maxWidth: '560px' }}>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '12px', textAlign: 'center', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  Try asking
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {EXAMPLE_QUERIES.map((q) => (
                    <button
                      key={q}
                      onClick={() => { setInput(q); textareaRef.current?.focus(); }}
                      style={{
                        padding: '12px 16px',
                        background: 'rgba(255,255,255,0.03)',
                        border: '1px solid rgba(255,255,255,0.07)',
                        borderRadius: '10px',
                        color: 'var(--text-secondary)',
                        fontSize: '13px',
                        cursor: 'pointer',
                        textAlign: 'left',
                        fontFamily: 'inherit',
                        transition: 'all 0.2s',
                      }}
                      onMouseEnter={(e) => {
                        (e.target as HTMLButtonElement).style.background = 'rgba(124,58,237,0.08)';
                        (e.target as HTMLButtonElement).style.color = 'var(--text-primary)';
                        (e.target as HTMLButtonElement).style.borderColor = 'rgba(124,58,237,0.2)';
                      }}
                      onMouseLeave={(e) => {
                        (e.target as HTMLButtonElement).style.background = 'rgba(255,255,255,0.03)';
                        (e.target as HTMLButtonElement).style.color = 'var(--text-secondary)';
                        (e.target as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.07)';
                      }}
                    >
                      {q} →
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))
        )}

        {/* Typing indicator */}
        {isLoading && (
          <div className="fade-in-up" style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
            <div style={{
              width: 28, height: 28, borderRadius: '8px', flexShrink: 0,
              background: 'linear-gradient(135deg, #0f172a, #1e293b)',
              border: '1px solid rgba(6,182,212,0.3)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '13px',
            }}>
              🤖
            </div>
            <div style={{
              padding: '14px 18px', borderRadius: '4px 16px 16px 16px',
              background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)',
              display: 'flex', alignItems: 'center', gap: '8px',
            }}>
              <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                {pipeline.stage === 'stage1_searching' && '🔍 Searching documents...'}
                {pipeline.stage === 'stage2_reranking' && '⚡ Reranking results...'}
                {pipeline.stage === 'llm_generating' && '✨ Generating answer...'}
              </span>
              <div style={{ display: 'flex', gap: '4px' }}>
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    style={{
                      width: 6, height: 6, borderRadius: '50%',
                      background: 'var(--text-muted)',
                      animation: `pulse 1.4s ease-in-out ${i * 0.2}s infinite`,
                    }}
                  />
                ))}
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div style={{
        padding: '16px 24px',
        background: 'rgba(8,13,26,0.8)',
        backdropFilter: 'blur(20px)',
        borderTop: '1px solid var(--glass-border)',
      }}>
        <div style={{
          display: 'flex', gap: '12px', alignItems: 'flex-end',
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: '14px',
          padding: '10px 14px',
          transition: 'border-color 0.2s',
        }}
          onFocus={() => {}}
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              !isInitialized
                ? 'Upload documents to start asking questions...'
                : 'Ask anything about your documents... (Enter to send, Shift+Enter for newline)'
            }
            disabled={isLoading || !isInitialized}
            rows={1}
            style={{
              flex: 1, background: 'transparent', border: 'none', outline: 'none',
              color: 'var(--text-primary)', fontSize: '14px', lineHeight: '1.6',
              fontFamily: 'inherit', resize: 'none', maxHeight: '140px',
              cursor: isInitialized ? 'text' : 'not-allowed',
            }}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim() || !isInitialized}
            style={{
              width: 38, height: 38, borderRadius: '10px', border: 'none',
              background: input.trim() && isInitialized && !isLoading
                ? 'linear-gradient(135deg, #7c3aed, #06b6d4)'
                : 'rgba(255,255,255,0.06)',
              color: 'white', cursor: input.trim() && isInitialized && !isLoading ? 'pointer' : 'not-allowed',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '16px', transition: 'all 0.2s', flexShrink: 0,
              boxShadow: input.trim() && isInitialized && !isLoading
                ? '0 0 20px rgba(124,58,237,0.4)' : 'none',
            }}
          >
            {isLoading ? <span className="spin" style={{ display: 'inline-block', fontSize: '14px' }}>⚙️</span> : '➤'}
          </button>
        </div>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '8px', textAlign: 'center' }}>
          Enter to send · Shift+Enter for new line · Answers grounded in your documents
        </div>
      </div>
    </div>
  );
};
