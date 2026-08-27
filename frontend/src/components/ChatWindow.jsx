import React, { useEffect, useRef } from 'react';
import { ArrowUpRight, ChevronDown, Loader2, Network, Send } from 'lucide-react';
import ResponseCard from './ResponseCard';

export default function ChatWindow({
  history,
  query,
  setQuery,
  onSubmit,
  isLoading,
  imagesCount,
  suggestedQueries = [],
  effort,
  setEffort,
  onOpenPipeline,
}) {
  const resultsRef = useRef(null);

  useEffect(() => {
    if (history.length && resultsRef.current) {
      resultsRef.current.scrollTo({ top: resultsRef.current.scrollHeight, behavior: 'smooth' });
    }
  }, [history, isLoading]);

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!isLoading && query.trim() && imagesCount) onSubmit();
    }
  };

  return (
    <div className="chat-window">
      <div className="chat-history-scroll" ref={resultsRef}>
        {isLoading && (
          <div className="loading-card">
            <Loader2 size={18} className="loading-orbit" />
            <div>
              <strong>Thinking<span className="thinking-dots" aria-hidden="true">...</span></strong>
              <p>OrbitLens is preparing your answer</p>
            </div>
          </div>
        )}

        {history.length > 0 && (
          <section className="answer-thread" aria-live="polite" aria-label="Analysis answers">
            {history.map((item, index) => (
              <ResponseCard
                key={`${item.query}-${index}`}
                responseData={item.response}
                queryText={item.query}
                answerOnly
                animate={index === history.length - 1}
              />
            ))}
          </section>
        )}
      </div>

      <section className="composer">
        <div className="composer-heading">
          <div>
            <span className="section-kicker">02 / Ask the scene</span>
            <h3>What do you want to know?</h3>
          </div>
          <span className="composer-label">{imagesCount ? `${imagesCount} tensor${imagesCount > 1 ? 's' : ''} ready` : 'Awaiting input'}</span>
        </div>

        {suggestedQueries.length > 0 && (
          <div className="suggested-wrap">
            <span className="suggested-label">Try a focused inquiry</span>
            <div className="suggestions">
              {suggestedQueries.map((suggestion) => (
                <button type="button" className="suggestion" key={suggestion} onClick={() => setQuery(suggestion)} disabled={isLoading}>
                  {suggestion} <ArrowUpRight size={12} style={{ verticalAlign: 'middle', marginLeft: 4 }} />
                </button>
              ))}
            </div>
          </div>
        )}

        <form className="query-form" onSubmit={(event) => { event.preventDefault(); if (!isLoading && query.trim() && imagesCount) onSubmit(); }}>
          <textarea
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={handleKeyDown}
            className="query-input"
            rows={2}
            disabled={isLoading || !imagesCount}
            placeholder={imagesCount
              ? 'Ask about change, location, objects, land cover, or sensor signals…'
              : 'Load one or two satellite images above to begin…'}
            aria-label="Satellite imagery question"
          />
          <div className="query-actions">
            <button type="button" className="tool-button" onClick={onOpenPipeline}>
              <Network size={13} /> Pipeline
            </button>
            <label className="effort-control">
              <span>Effort</span>
              <select value={effort} onChange={(event) => setEffort(event.target.value)} disabled={isLoading}>
                <option value="min">Min</option>
                <option value="medium">Medium</option>
                <option value="max">Max</option>
              </select>
              <ChevronDown size={12} />
            </label>
            <button type="submit" className="primary-button" disabled={isLoading || !query.trim() || !imagesCount}>
              {isLoading ? <><Loader2 size={14} className="loading-orbit" /> Analyzing</> : <>Run analysis <Send size={13} /></>}
            </button>
          </div>
        </form>
        <div className="composer-helper">
          <span><kbd>Enter</kbd> to run · <kbd>Shift + Enter</kbd> for a new line</span>
          {!imagesCount && <span className="helper-warning">Image input required</span>}
        </div>
      </section>
    </div>
  );
}