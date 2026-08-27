import React from 'react';
import { Cpu, Info, Satellite } from 'lucide-react';

export default function Header({ backendHealth, onOpenArchitecture }) {
  const isHealthy = backendHealth?.status === 'ok';
  const activeModel = backendHealth?.active_model || 'Connecting to engine';

  return (
    <header className="topbar">
      <div className="topbar-inner">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true"><Satellite size={18} /></div>
          <div className="brand-copy">
            <span className="brand-name">OrbitLens</span>
            <span className="brand-sub">Earth observation / 01</span>
          </div>
        </div>

        <div className="topbar-actions">
          <div className="system-status" title={activeModel}>
            <span className={`status-dot ${isHealthy ? 'is-on' : ''}`} />
            <span>{isHealthy ? 'Engine ready' : 'Engine connecting'}</span>
            <Cpu size={13} />
            <span className="status-model">{activeModel}</span>
          </div>
          <button type="button" className="outline-button" onClick={onOpenArchitecture}>
            <Info size={14} />
            <span>System map</span>
          </button>
        </div>
      </div>
    </header>
  );
}