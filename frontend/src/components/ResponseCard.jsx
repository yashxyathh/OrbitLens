import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, Check, ChevronDown, Copy, Gauge, ShieldCheck, User } from 'lucide-react';
import TraceViewer from './TraceViewer';

const taskLabels = {
  change_detection: 'Bi-temporal change detection',
  optical_sar_fusion: 'Optical + SAR sensor fusion',
  grounding: 'Spatial visual grounding',
  captioning: 'Satellite scene captioning',
  single_image_vqa: 'Remote sensing visual Q&A',
};

function TypingAnswer({ answer, animate }) {
  const [visibleLength, setVisibleLength] = useState(animate ? 0 : answer.length);

  useEffect(() => {
    if (!animate || !answer) {
      setVisibleLength(answer?.length || 0);
      return undefined;
    }

    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      setVisibleLength(answer.length);
      return undefined;
    }

    setVisibleLength(0);
    let currentLength = 0;
    const increment = Math.max(1, Math.ceil(answer.length / 110));
    const timer = window.setInterval(() => {
      currentLength = Math.min(answer.length, currentLength + increment);
      setVisibleLength(currentLength);
      if (currentLength >= answer.length) window.clearInterval(timer);
    }, 22);

    return () => window.clearInterval(timer);
  }, [answer, animate]);

  const isComplete = visibleLength >= answer.length;
  if (!animate || isComplete) {
    return <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>;
  }

  return (
    <span className="typing-text">
      {answer.slice(0, visibleLength)}
      <span className="typing-caret" aria-hidden="true" />
    </span>
  );
}

export default function ResponseCard({ responseData, queryText, answerOnly = false, animate = false }) {
  const [copied, setCopied] = useState(false);
  const [isAnalysisOpen, setIsAnalysisOpen] = useState(false);
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

  if (answerOnly) {
    return (
      <article className={`response-card response-card-answer-only ${isAnalysisOpen ? 'is-analysis-open' : ''}`}>
        <div className="response-body">
          <div className="prose-sat">
            <TypingAnswer answer={answer || ''} animate={animate} />
          </div>
        </div>
        <div className="answer-actions">
          <button
            type="button"
            className="analysis-toggle"
            onClick={() => setIsAnalysisOpen((isOpen) => !isOpen)}
            aria-expanded={isAnalysisOpen}
          >
            <span><ChevronDown size={14} /> Analysis log</span>
            <small>{isAnalysisOpen ? 'Hide' : 'Show'}</small>
          </button>
          <button type="button" className="copy-button" onClick={handleCopy}>
            {copied ? <Check size={13} /> : <Copy size={13} />} {copied ? 'Copied' : 'Copy'}
          </button>
        </div>

        {isAnalysisOpen && (
          <div className="answer-analysis">
            <div className="response-top">
              <span className="task-label">{taskLabels[taskType] || taskType || 'Specialist assessment'}</span>
              <span className="confidence"><Gauge size={13} style={{ verticalAlign: 'middle', marginRight: 5 }} /> {confidencePercent}%</span>
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
            <div className="response-note">
              <ShieldCheck size={13} />
              <span>{confidenceLabel || 'Confidence is calibrated from routing and model certainty.'}</span>
            </div>
            {trace && <TraceViewer trace={trace} />}
          </div>
        )}
      </article>
    );
  }

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
           <TypingAnswer answer={answer || ''} animate={animate} />
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