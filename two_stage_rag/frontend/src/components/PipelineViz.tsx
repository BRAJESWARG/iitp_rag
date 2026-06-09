// ─────────────────────────────────────────────────────────────────
// components/PipelineViz.tsx — Animated pipeline stage diagram
// ─────────────────────────────────────────────────────────────────
import React from 'react';
import { PipelineState } from '../types';

interface PipelineVizProps {
  pipeline: PipelineState;
}

interface Stage {
  id: string;
  icon: string;
  title: string;
  subtitle: string;
  color: string;
  glow: string;
  activeStage: PipelineState['stage'];
}

const stages: Stage[] = [
  {
    id: 'stage1',
    icon: '🔍',
    title: 'Hybrid Search',
    subtitle: 'BM25 + Vector',
    color: '#3b82f6',
    glow: 'rgba(59,130,246,0.35)',
    activeStage: 'stage1_searching',
  },
  {
    id: 'stage2',
    icon: '⚡',
    title: 'Cross Encoder',
    subtitle: 'Reranker → Top 5',
    color: '#8b5cf6',
    glow: 'rgba(139,92,246,0.35)',
    activeStage: 'stage2_reranking',
  },
  {
    id: 'llm',
    icon: '✨',
    title: 'Gemini LLM',
    subtitle: 'gemini-2.5-flash',
    color: '#f59e0b',
    glow: 'rgba(245,158,11,0.35)',
    activeStage: 'llm_generating',
  },
];

export const PipelineViz: React.FC<PipelineVizProps> = ({ pipeline }) => {
  const isActive = (s: Stage) => pipeline.stage === s.activeStage;
  const isDone = pipeline.stage === 'done';

  const stageIndex = stages.findIndex((s) => pipeline.stage === s.activeStage);

  return (
    <div
      className="glass-card"
      style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          Pipeline
        </span>
        {pipeline.stage !== 'idle' && (
          <span style={{ fontSize: '11px', color: pipeline.stage === 'error' ? 'var(--error)' : pipeline.stage === 'done' ? 'var(--success)' : 'var(--warning)' }}>
            {pipeline.stage === 'done' ? '✓ Complete' : pipeline.stage === 'error' ? '✗ Error' : 'Running...'}
          </span>
        )}
      </div>

      {/* Stages */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {stages.map((stage, idx) => {
          const active = isActive(stage);
          const done = isDone || stageIndex > idx;

          return (
            <React.Fragment key={stage.id}>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  padding: '10px 12px',
                  borderRadius: '10px',
                  transition: 'all 0.3s ease',
                  background: active
                    ? `${stage.color}12`
                    : done
                    ? 'rgba(16,185,129,0.07)'
                    : 'transparent',
                  border: `1px solid ${active ? stage.color + '40' : done ? 'rgba(16,185,129,0.2)' : 'transparent'}`,
                  boxShadow: active ? `0 0 16px ${stage.glow}` : 'none',
                }}
              >
                {/* Icon */}
                <div
                  style={{
                    width: 34, height: 34, borderRadius: '8px',
                    background: active ? stage.color : done ? 'rgba(16,185,129,0.2)' : 'rgba(255,255,255,0.05)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '15px', flexShrink: 0,
                    transition: 'all 0.3s ease',
                    boxShadow: active ? `0 0 12px ${stage.glow}` : 'none',
                  }}
                >
                  {active ? (
                    <span className="spin" style={{ display: 'inline-block', fontSize: '14px' }}>⚙️</span>
                  ) : done ? (
                    '✓'
                  ) : (
                    stage.icon
                  )}
                </div>

                {/* Labels */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{
                    fontSize: '13px', fontWeight: 600,
                    color: active ? stage.color : done ? 'var(--success)' : 'var(--text-secondary)',
                    transition: 'color 0.3s',
                  }}>
                    {stage.title}
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    {stage.subtitle}
                  </div>
                </div>

                {/* Count badge */}
                {done && stage.id === 'stage1' && pipeline.stage1Count !== undefined && (
                  <span className="badge badge-info" style={{ fontSize: '11px' }}>
                    {pipeline.stage1Count} docs
                  </span>
                )}
                {done && stage.id === 'stage2' && pipeline.stage2Count !== undefined && (
                  <span className="badge badge-purple" style={{ fontSize: '11px' }}>
                    Top {pipeline.stage2Count}
                  </span>
                )}
              </div>

              {/* Connector arrow */}
              {idx < stages.length - 1 && (
                <div style={{
                  display: 'flex', justifyContent: 'center',
                  color: stageIndex > idx || isDone ? 'var(--success)' : 'var(--text-muted)',
                  fontSize: '13px', lineHeight: 1,
                  transition: 'color 0.3s',
                }}>
                  ↓
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Progress bar when active */}
      {pipeline.stage !== 'idle' && pipeline.stage !== 'done' && pipeline.stage !== 'error' && (
        <div style={{ height: '2px', background: 'var(--glass-border)', borderRadius: '2px', overflow: 'hidden' }}>
          <div className="progress-bar" key={pipeline.stage} />
        </div>
      )}
    </div>
  );
};
