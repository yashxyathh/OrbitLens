import React, { useEffect, useRef } from 'react';
import { ArrowUpRight, Loader2, Send, Trash2 } from 'lucide-react';
import ResponseCard from './ResponseCard';

export default function ChatWindow({
  history,
  query,
  setQuery,
  onSubmit,
  isLoading,
  onClearHistory,
  imagesCount,
  suggestedQueries = [],
}) {
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (history.length) messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history, isLoading]);

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!isLoading && query.trim() && imagesCount) onSubmit();
    }
  };

  return (
    <div>
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
            rows={4}
            disabled={isLoading || !imagesCount}
            placeholder={imagesCount
              ? 'Ask about change, location, objects, land cover, or sensor signals…'
              : 'Load one or two satellite images above to begin…'}
            aria-label="Satellite imagery question"
          />
          <div className="query-actions">
            {history.length > 0 && (
              <button type="button" className="clear-history" onClick={onClearHistory} disabled={isLoading} aria-label="Clear analysis history">
                <Trash2 size={14} />
              </button>
            )}
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

      {isLoading && (
        <div className="loading-card">
          <Loader2 size={22} className="loading-orbit" />
          <div>
            <strong>Reading the scene</strong>
            <p>Validating tensors · routing specialist · assembling evidence</p>
          </div>
        </div>
      )}

      {history.length > 0 && (
        <section className="results" aria-live="polite">
          <div className="results-heading">
            <h3>Analysis log</h3>
            <span>{history.length} {history.length === 1 ? 'inquiry' : 'inquiries'}</span>
          </div>
          {history.map((item, index) => (
            <ResponseCard key={`${item.query}-${index}`} responseData={item.response} queryText={item.query} imagePreviews={item.imagePreviews} />
          ))}
          <div ref={messagesEndRef} />
        </section>
      )}
    </div>
  );
}