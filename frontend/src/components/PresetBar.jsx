import React from 'react';
import { Layers, Zap, ArrowRight } from 'lucide-react';

export default function PresetBar({ presets, activePresetId, onSelectPreset }) {
  if (!presets || presets.length === 0) return null;

  return (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-2.5">
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
            Quick Evaluation Presets (1-Click Judge Demos)
          </span>
        </div>
        <span className="text-[11px] text-slate-400 font-mono hidden sm:inline">
          Preloaded with high-res optical & SAR satellite benchmarks
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2.5">
        {presets.map((preset) => {
          const isSelected = activePresetId === preset.id;
          return (
            <button
              key={preset.id}
              onClick={() => onSelectPreset(preset)}
              className={`text-left p-2.5 rounded-xl transition-all duration-200 border relative group overflow-hidden ${
                isSelected
                  ? 'bg-cyan-950/60 border-cyan-500 shadow-md shadow-cyan-500/10'
                  : 'bg-slate-900/80 hover:bg-slate-800/90 border-slate-800 hover:border-slate-700'
              }`}
            >
              {/* Category Pill */}
              <div className="flex items-center justify-between mb-1.5">
                <span className={`text-[10px] font-mono font-medium px-2 py-0.5 rounded-md ${
                  preset.category === 'change_detection' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' :
                  preset.category === 'optical_sar_fusion' ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30' :
                  preset.category === 'grounding' ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' :
                  preset.category === 'captioning' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' :
                  'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                }`}>
                  {preset.category.replace('_', ' ').toUpperCase()}
                </span>
                <span className="text-[10px] font-mono text-slate-400">
                  {preset.image_names.length} {preset.image_names.length === 1 ? 'img' : 'imgs'}
                </span>
              </div>

              {/* Title */}
              <h4 className="text-xs font-semibold text-slate-200 group-hover:text-cyan-300 transition-colors line-clamp-1 mb-1">
                {preset.title}
              </h4>

              {/* Description */}
              <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                {preset.description}
              </p>

              {/* Selection indicator line */}
              {isSelected && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-cyan-400 to-indigo-500" />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
