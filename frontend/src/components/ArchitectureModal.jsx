import React from 'react';
import { ArrowRight, Database, Network, X } from 'lucide-react';

const stages = [
  ['01', 'Query + tensor ingestion', 'Natural language plus one or two optical, SAR, or bi-temporal images.'],
  ['02', 'Compatibility validation', 'PIL integrity checks, image dimensions, and payload limits.'],
  ['03', 'Agentic task router', 'Classifies the inquiry into five remote sensing specialist tasks.'],
  ['04', 'Specialist conditioning', 'Applies a domain prompt and runs the configured vision model.'],
  ['05', 'Evidence response', 'Returns observations, confidence calibration, and an auditable trace.'],
];

const roadmap = [
  ['BigEarthNet-MM', 'Multi-spectral optical and Sentinel-1 SAR classification.'],
  ['VRSBench + RSICD', 'Scene captioning, visual grounding, and localization.'],
  ['RSVQA', 'Visual Q&A for counts, infrastructure, and spatial relations.'],
  ['CDVQA', 'Bi-temporal change detection for damage and urban growth.'],
];

export default function ArchitectureModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-panel" onClick={(event) => event.stopPropagation()}>
        <button type="button" className="modal-close" onClick={onClose} aria-label="Close system map"><X size={16} /></button>
        <div className="modal-heading">
          <Network size={24} />
          <div>
            <h2>How SatQuery thinks</h2>
            <p>Five stages from image to intelligence · SIH 2026 / ISRO</p>
          </div>
        </div>

        <section>
          <div className="modal-kicker">The agentic pipeline</div>
          <div className="stage-grid">
            {stages.map(([number, title, description]) => (
              <div className="stage-card" key={number}>
                <b>{number}</b>
                <h4>{title}</h4>
                <p>{description}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="roadmap">
          <div className="modal-kicker"><Database size={13} style={{ verticalAlign: 'middle', marginRight: 5 }} /> Ready for specialist models</div>
          <div className="roadmap-grid">
            {roadmap.map(([title, description]) => (
              <div className="roadmap-card" key={title}>
                <b>{title}</b>
                <p style={{ marginTop: 9 }}>{description}</p>
              </div>
            ))}
          </div>
        </section>

        <div className="modal-footer">
          <button type="button" className="primary-button" onClick={onClose}>Return to workspace <ArrowRight size={14} /></button>
        </div>
      </div>
    </div>
  );
}