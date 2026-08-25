import React from 'react';
import { X, Layers, Network, Cpu, Database, CheckCircle, ArrowRight, ShieldCheck, Sparkles } from 'lucide-react';

export default function ArchitectureModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-space-900 border border-slate-700 rounded-2xl overflow-y-auto shadow-2xl p-6 sm:p-8">
        
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-6 right-6 p-2 rounded-xl bg-slate-800 text-slate-300 hover:text-white border border-slate-700 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Modal Header */}
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-cyan-950 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
            <Network className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-100 font-display">
              SatQuery AI — 5-Stage Agentic Architecture
            </h2>
            <p className="text-xs text-cyan-400 font-mono">
              ISRO Problem Statement • Smart India Hackathon (SIH 2026) System Blueprint
            </p>
          </div>
        </div>

        {/* 5 Stages Breakdown */}
        <div className="space-y-4 mb-8">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 font-mono">
            Pipeline Execution Stages
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
            {[
              {
                num: "01",
                title: "Query & Tensor Ingestion",
                desc: "Accepts natural language text + 1 or 2 satellite images (optical, SAR, or bi-temporal pair)."
              },
              {
                num: "02",
                title: "Compatibility Validation",
                desc: "PIL-based integrity verification, dimensions check, and rejection of corrupt or >2 image payloads."
              },
              {
                num: "03",
                title: "Agentic Task Router",
                desc: "Classifies intent into 5 RS specialist tasks: RSVQA, Captioning, Grounding, CDVQA, or SAR Fusion."
              },
              {
                num: "04",
                title: "Specialist Conditioning",
                desc: "Custom prompt engineering + Vision-LLM execution (Anthropic Claude 3.7 Vision API / Domain Engine)."
              },
              {
                num: "05",
                title: "Evidence Response",
                desc: "Delivers markdown observations, calibrated confidence meter, and auditable step-by-step execution trace."
              }
            ].map((st, i) => (
              <div key={i} className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 flex flex-col justify-between">
                <div>
                  <span className="text-cyan-400 font-mono font-bold text-xs">{st.num}</span>
                  <h4 className="text-xs font-semibold text-slate-200 mt-1 mb-1">{st.title}</h4>
                  <p className="text-[11px] text-slate-400 leading-snug">{st.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Future Model Swapping & Datasets Roadmap */}
        <div className="p-5 rounded-2xl bg-gradient-to-br from-indigo-950/40 via-slate-950/60 to-cyan-950/40 border border-indigo-500/30 mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Database className="w-5 h-5 text-indigo-400" />
            <h3 className="text-sm font-bold text-slate-100 font-display">
              Fine-Tuning Integration Seams (Downstream Model Swapping)
            </h3>
          </div>
          <p className="text-xs text-slate-300 mb-4 leading-relaxed">
            SatQuery AI is designed with modular adapter interfaces in <code className="text-cyan-300 font-mono">llm_client.py</code>. During post-hackathon scaling, the vision-LLM endpoints will be directly replaced with dedicated PyTorch/HuggingFace checkpoints fine-tuned on benchmark remote sensing corpora:
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
            <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
              <span className="text-cyan-400 font-mono font-bold block mb-1">BigEarthNet-MM</span>
              <p className="text-[11px] text-slate-400">Multi-spectral optical & Sentinel-1 SAR classification and land cover segmentation.</p>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
              <span className="text-indigo-400 font-mono font-bold block mb-1">VRSBench & RSICD</span>
              <p className="text-[11px] text-slate-400">Satellite scene captioning, visual grounding, and high-resolution spatial localization.</p>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
              <span className="text-emerald-400 font-mono font-bold block mb-1">RSVQA (High/Low Res)</span>
              <p className="text-[11px] text-slate-400">Visual question answering for object counts, infrastructure, and geospatial relations.</p>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
              <span className="text-amber-400 font-mono font-bold block mb-1">CDVQA (Bi-Temporal)</span>
              <p className="text-[11px] text-slate-400">Change detection question answering for flood, disaster damage, and urban sprawl tracking.</p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs shadow-lg shadow-cyan-600/20 transition-all"
          >
            Close Blueprint
          </button>
        </div>

      </div>
    </div>
  );
}
