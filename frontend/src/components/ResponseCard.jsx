import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, User, Check, Copy, Shield, Gauge, Sparkles, AlertTriangle, Layers } from 'lucide-react';
import TraceViewer from './TraceViewer';

export default function ResponseCard({ responseData, queryText, imagePreviews = [] }) {
  const [copied, setCopied] = useState(false);

  if (!responseData) return null;

  const { answer, confidence, confidence_label, task_type, trace } = responseData;

  const handleCopy = () => {
    navigator.clipboard.writeText(answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Confidence color grading
  const confidencePercent = Math.round(confidence * 100);
  const confidenceColor = 
    confidencePercent >= 90 ? 'text-emerald-400 border-emerald-500/40 bg-emerald-950/40' :
    confidencePercent >= 75 ? 'text-cyan-400 border-cyan-500/40 bg-cyan-950/40' :
    'text-amber-400 border-amber-500/40 bg-amber-950/40';

  const taskBadges = {
    change_detection: {
      label: 'Bi-Temporal Change Detection (CDVQA)',
      color: 'bg-amber-950/80 text-amber-300 border-amber-800/80',
      icon: '🔄'
    },
    optical_sar_fusion: {
      label: 'Multi-Sensor Optical + SAR Fusion',
      color: 'bg-indigo-950/80 text-indigo-300 border-indigo-800/80',
      icon: '🛰️'
    },
    grounding: {
      label: 'Spatial Visual Grounding & Localization',
      color: 'bg-rose-950/80 text-rose-300 border-rose-800/80',
      icon: '🎯'
    },
    captioning: {
      label: 'Satellite Scene Captioning (VRSBench)',
      color: 'bg-emerald-950/80 text-emerald-300 border-emerald-800/80',
      icon: '📝'
    },
    single_image_vqa: {
      label: 'Remote Sensing VQA (RSVQA)',
      color: 'bg-cyan-950/80 text-cyan-300 border-cyan-800/80',
      icon: '🔍'
    }
  };

  const badgeInfo = taskBadges[task_type] || {
    label: task_type,
    color: 'bg-slate-800 text-slate-300 border-slate-700',
    icon: '🛰️'
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-2xl space-y-4">
      
      {/* Top Bar: Task Type Badge & Confidence Meter */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
        
        {/* Task Badge */}
        <div className="flex items-center gap-2">
          <div className={`px-3 py-1 rounded-lg border text-xs font-semibold flex items-center gap-1.5 shadow-sm ${badgeInfo.color}`}>
            <span>{badgeInfo.icon}</span>
            <span>{badgeInfo.label}</span>
          </div>
        </div>

        {/* Confidence Indicator */}
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3 py-1 rounded-lg border text-xs font-mono ${confidenceColor}`}>
            <Gauge className="w-3.5 h-3.5" />
            <span className="font-bold">{confidencePercent}%</span>
            <span className="text-[11px] opacity-80 hidden sm:inline">({confidence_label})</span>
          </div>

          {/* Copy Button */}
          <button
            onClick={handleCopy}
            className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition-all text-xs flex items-center gap-1"
            title="Copy response markdown"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span className="hidden md:inline font-mono">{copied ? 'Copied' : 'Copy'}</span>
          </button>
        </div>

      </div>

      {/* User Query Echo */}
      {queryText && (
        <div className="flex items-start gap-3 p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
          <div className="w-6 h-6 rounded-md bg-cyan-950 border border-cyan-800/60 flex items-center justify-center flex-shrink-0 text-cyan-400 mt-0.5">
            <User className="w-3.5 h-3.5" />
          </div>
          <div className="flex-1">
            <span className="text-[11px] font-mono text-cyan-400 font-semibold block mb-0.5">
              Input Query & Task Instruction
            </span>
            <p className="text-sm text-slate-200 font-medium">
              {queryText}
            </p>
          </div>
        </div>
      )}

      {/* Model Answer Body */}
      <div className="p-4 rounded-xl bg-slate-950/40 border border-slate-800/60">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-6 h-6 rounded-md bg-indigo-950 border border-indigo-800/60 flex items-center justify-center text-indigo-400">
            <Bot className="w-3.5 h-3.5" />
          </div>
          <span className="text-xs font-semibold uppercase tracking-wider text-indigo-300 font-display">
            Specialist Remote Sensing Assessment
          </span>
        </div>

        <div className="prose prose-invert prose-sat max-w-none text-sm">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {answer}
          </ReactMarkdown>
        </div>

        {/* Confidence disclaimer note */}
        <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center gap-2 text-[11px] text-slate-500 font-mono">
          <Shield className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
          <span>
            * Confidence score is calibrated via specialist routing heuristics & vision token certainty (ready for fine-tuned logits integration).
          </span>
        </div>
      </div>

      {/* Collapsible Agentic Trace Viewer */}
      {trace && <TraceViewer trace={trace} />}

    </div>
  );
}
