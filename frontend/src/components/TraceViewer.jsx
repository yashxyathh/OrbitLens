import React, { useState } from 'react';
import { Terminal, ChevronDown, ChevronUp, CheckCircle2, Cpu, Clock, Layers, Route, Sparkles } from 'lucide-react';

export default function TraceViewer({ trace }) {
  const [isOpen, setIsOpen] = useState(true);

  if (!trace) return null;

  return (
    <div className="mt-4 border border-slate-800 bg-slate-950/60 rounded-xl overflow-hidden shadow-inner font-mono text-xs">
      
      {/* Trace Toggle Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-2.5 bg-slate-900/90 hover:bg-slate-900 border-b border-slate-800 flex items-center justify-between transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-cyan-400" />
          <span className="font-semibold text-slate-200 uppercase tracking-wider text-[11px]">
            Agentic Execution Trace & Telemetry
          </span>
          <span className="px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800/50 text-[10px]">
            {trace.task_type}
          </span>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 text-slate-400 text-[11px]">
            <Clock className="w-3.5 h-3.5 text-amber-400" />
            <span>{trace.execution_time_ms} ms</span>
          </div>
          {isOpen ? (
            <ChevronUp className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          )}
        </div>
      </button>

      {/* Expanded Pipeline Steps */}
      {isOpen && (
        <div className="p-4 space-y-3.5 bg-slate-950/80">
          
          {/* Top metadata summary row */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 pb-3 border-b border-slate-800/80 text-[11px]">
            <div className="flex items-center gap-1.5 text-slate-400">
              <Route className="w-3.5 h-3.5 text-indigo-400" />
              <span>Routing Pipeline:</span>
              <span className="text-indigo-300 font-semibold">{trace.task_type}</span>
            </div>
            <div className="flex items-center gap-1.5 text-slate-400">
              <Cpu className="w-3.5 h-3.5 text-emerald-400" />
              <span>Inference Engine:</span>
              <span className="text-emerald-300 font-semibold truncate max-w-[140px]" title={trace.model_invoked}>
                {trace.model_invoked}
              </span>
            </div>
            <div className="flex items-center gap-1.5 text-slate-400">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              <span>Input Tensors:</span>
              <span className="text-cyan-300 font-semibold">{trace.image_count} satellite tensor(s)</span>
            </div>
          </div>

          {/* Sequential step timeline */}
          <div className="relative pl-6 space-y-3 before:absolute before:left-2.5 before:top-2 before:bottom-2 before:w-0.5 before:bg-slate-800">
            {trace.pipeline_steps?.map((step, idx) => (
              <div key={idx} className="relative group">
                {/* Step Node Marker */}
                <div className="absolute -left-6 top-0.5 w-5 h-5 rounded-full bg-slate-900 border border-cyan-500/50 flex items-center justify-center text-[10px] text-cyan-400 shadow-sm">
                  {idx + 1}
                </div>

                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-slate-200">
                      {step.stage}
                    </span>
                    <span className="text-[10px] text-emerald-400 bg-emerald-950/60 border border-emerald-800/40 px-1.5 rounded">
                      {step.status}
                    </span>
                  </div>

                  <p className="text-slate-300 mt-0.5 leading-relaxed text-[11px]">
                    {step.description}
                  </p>

                  {step.details && (
                    <div className="mt-1 p-2 rounded bg-slate-900/90 border border-slate-800 text-[10.5px] text-slate-400 break-words leading-relaxed font-sans">
                      <span className="text-cyan-400 font-mono font-medium">Log Details: </span>
                      {step.details}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Routing Reason Callout */}
          {trace.routing_reason && (
            <div className="mt-3 p-2.5 rounded-lg bg-indigo-950/40 border border-indigo-800/50 text-[11px] text-indigo-200">
              <span className="font-semibold text-indigo-300 font-mono">Agentic Routing Rationale: </span>
              {trace.routing_reason}
            </div>
          )}

        </div>
      )}

    </div>
  );
}
