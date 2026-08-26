import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, Check, Copy, Gauge, ShieldCheck, User } from 'lucide-react';
import TraceViewer from './TraceViewer';

const taskLabels = {
  change_detection: 'Bi-temporal change detection',
  optical_sar_fusion: 'Optical + SAR sensor fusion',
  grounding: 'Spatial visual grounding',
  captioning: 'Satellite scene captioning',
  single_image_vqa: 'Remote sensing visual Q&A',
};

export default function ResponseCard({ responseData, queryText }) {
  const [copied, setCopied] = useState(false);
  if (!responseData) return null;

  const { answer, confidence, confidence_label: confidenceLabel, task_type: taskType, trace } = responseData;
  const confidencePercent = Math.round((confidence || 0) * 100);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(answer || '');
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };

  return (
    <article className="response-card">
      <div className="response-top">
        <span className="task-label">{taskLabels[taskType] || taskType || 'Specialist assessment'}</span>
        <div className="response-tools">
          <span className="confidence"><Gauge size={13} style={{ verticalAlign: 'middle', marginRight: 5 }} /> {confidencePercent}%</span>
          <button type="button" className="copy-button" onClick={handleCopy}>
            {copied ? <Check size={13} /> : <Copy size={13} />} {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      </div>

      {queryText && (
        <div className="query-echo">
          <User size={15} />
          <div>
            <span>Your inquiry</span>
            <p>{queryText}</p>
          </div>
        </div>
      )}

      <div className="response-body">
        <div className="response-body-heading"><Bot size={15} /> Specialist assessment</div>
        <div className="prose-sat">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
        </div>
        <div className="response-note">
          <ShieldCheck size={13} />
          <span>{confidenceLabel || 'Confidence is calibrated from routing and model certainty.'}</span>
        </div>
      </div>

      {trace && <TraceViewer trace={trace} />}
    </article>
  );
}