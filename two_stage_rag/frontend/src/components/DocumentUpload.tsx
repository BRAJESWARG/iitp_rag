// ─────────────────────────────────────────────────────────────────
// components/DocumentUpload.tsx — Drag & drop file ingestion
// ─────────────────────────────────────────────────────────────────
import React, { useCallback, useState, useRef } from 'react';
import { IngestResponse } from '../types';

interface DocumentUploadProps {
  onIngest: (files: File[]) => Promise<IngestResponse>;
  isIngesting: boolean;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({ onIngest, isIngesting }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const arr = Array.from(files).filter((f) =>
      f.name.endsWith('.pdf') || f.name.endsWith('.txt')
    );
    if (arr.length === 0) {
      setError('Only PDF and TXT files are supported.');
      return;
    }
    setError(null);
    setResult(null);
    try {
      const res = await onIngest(arr);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Ingestion failed.');
    }
  }, [onIngest]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  return (
    <div className="glass-card" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        Documents
      </span>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${isDragging ? 'var(--grad-start)' : 'rgba(255,255,255,0.12)'}`,
          borderRadius: '10px',
          padding: '20px 16px',
          textAlign: 'center',
          cursor: isIngesting ? 'not-allowed' : 'pointer',
          transition: 'all 0.2s ease',
          background: isDragging ? 'rgba(124,58,237,0.08)' : 'transparent',
          boxShadow: isDragging ? '0 0 20px rgba(124,58,237,0.15)' : 'none',
        }}
      >
        <div style={{ fontSize: '28px', marginBottom: '8px' }}>
          {isIngesting ? <span className="spin" style={{ display: 'inline-block' }}>⚙️</span> : '📄'}
        </div>
        <div style={{ fontSize: '13px', color: 'var(--text-primary)', fontWeight: 500 }}>
          {isIngesting ? 'Ingesting...' : 'Drop files here'}
        </div>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
          PDF or TXT • Click to browse
        </div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.txt"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
          disabled={isIngesting}
        />
      </div>

      {/* Result */}
      {result && (
        <div style={{
          padding: '10px 12px', borderRadius: '8px',
          background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.2)',
        }}>
          <div style={{ fontSize: '12px', color: 'var(--success)', fontWeight: 600, marginBottom: '4px' }}>
            ✓ {result.message}
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            {result.total_chunks} chunks indexed
          </div>
          <div style={{ marginTop: '6px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
            {result.files_ingested.map((f) => (
              <span key={f} style={{
                fontSize: '10px', padding: '2px 8px', borderRadius: '20px',
                background: 'rgba(16,185,129,0.15)', color: 'var(--success)',
              }}>
                {f}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          padding: '10px 12px', borderRadius: '8px',
          background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)',
          fontSize: '12px', color: 'var(--error)',
        }}>
          ⚠️ {error}
        </div>
      )}
    </div>
  );
};
