import React from 'react';
import { Satellite, ShieldCheck, Activity, Cpu, Info, Sparkles } from 'lucide-react';

export default function Header({ backendHealth, onOpenArchitecture }) {
  const isHealthy = backendHealth?.status === 'ok';
  const activeModel = backendHealth?.active_model || 'Connecting...';
  const isLiveClaude = activeModel.toLowerCase().includes('anthropic') || activeModel.toLowerCase().includes('claude');

  return (
    <header className="border-b border-slate-800 bg-space-900/80 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Logo and Title */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 shadow-lg shadow-cyan-500/20">
            <Satellite className="w-5 h-5 text-white animate-pulse" />
            <div className="absolute inset-0 rounded-xl border border-cyan-300/40 animate-ping opacity-25" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight font-display bg-gradient-to-r from-white via-cyan-100 to-cyan-400 bg-clip-text text-transparent">
                SatQuery AI
              </h1>
              <span className="px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800/60 flex items-center gap-1">
                <Sparkles className="w-2.5 h-2.5" /> SIH 2026 / ISRO
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Agentic Vision-Language Assistant for Remote Sensing & Earth Observation
            </p>
          </div>
        </div>

        {/* Right Action & Status Pills */}
        <div className="flex items-center gap-2 sm:gap-4">
          
          {/* Architecture Details Modal Trigger */}
          <button
            onClick={onOpenArchitecture}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-slate-800/70 hover:bg-slate-700/80 border border-slate-700 rounded-lg transition-all"
            title="View 5-Stage Agentic Architecture & Dataset Roadmap"
          >
            <Info className="w-3.5 h-3.5 text-cyan-400" />
            <span className="hidden md:inline">Architecture & Roadmap</span>
          </button>

          {/* Backend Status Indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <div className="flex items-center gap-1.5">
              <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' : 'bg-rose-400 animate-ping'}`} />
              <span className="text-slate-300 font-mono hidden lg:inline">Backend:</span>
            </div>
            <div className="flex items-center gap-1">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              <span className={`font-mono font-medium truncate max-w-[140px] sm:max-w-[200px] ${isLiveClaude ? 'text-cyan-300' : 'text-emerald-300'}`}>
                {activeModel}
              </span>
            </div>
          </div>

        </div>

      </div>
    </header>
  );
}
