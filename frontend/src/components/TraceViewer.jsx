import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Clock3, Cpu, Layers3, Route, Terminal } from 'lucide-react';

export default function TraceViewer({ trace }) {
  const [isOpen, setIsOpen] = useState(false);
  if (!trace) return null;

  return (
    <div className="trace">
      <button type="button" className="trace-toggle" onClick={() => setIsOpen((open) => !open)}>
        <span className="trace-toggle-title"><Terminal size={14} /> Execution trace</span>
        <span className="trace-toggle-meta"><span><Clock3 size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />{trace.execution_time_ms} ms</span>{isOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</span>
      </button>

      {isOpen && (
        <div className="trace-content">
          <div className="trace-meta">
            <div><span><Route size={11} style={{ verticalAlign: 'middle', marginRight: 4 }} />Route</span><strong>{trace.task_type}</strong></div>
            <div><span><Cpu size={11} style={{ verticalAlign: 'middle', marginRight: 4 }} />Engine</span><strong title={trace.model_invoked}>{trace.model_invoked}</strong></div>
            <div><span><Layers3 size={11} style={{ verticalAlign: 'middle', marginRight: 4 }} />Inputs</span><strong>{trace.image_count} tensor(s)</strong></div>
          </div>
          <ol className="trace-steps">
            {trace.pipeline_steps?.map((step, index) => (
              <li className="trace-step" key={`${step.stage}-${index}`}>
                <span className="trace-number">{String(index + 1).padStart(2, '0')}</span>
                <div>
                  <div className="trace-step-head"><strong>{step.stage}</strong><span>{step.status || 'completed'}</span></div>
                  <p>{step.description}</p>
                  {step.details && <div className="trace-detail"><b>Log detail · </b>{step.details}</div>}
                </div>
              </li>
            ))}
          </ol>
          {trace.routing_reason && <div className="trace-callout"><b>Routing rationale · </b>{trace.routing_reason}</div>}
        </div>
      )}
    </div>
  );
}