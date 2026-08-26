import React, { useRef, useEffect } from 'react';
import { Send, Sparkles, Loader2, MessageSquare, Trash2, ArrowUpRight } from 'lucide-react';
import ResponseCard from './ResponseCard';

export default function ChatWindow({
  history,
  query,
  setQuery,
  effort,
  setEffort,
  onSubmit,
  isLoading,
  onClearHistory,
  imagesCount,
  suggestedQueries = []
}) {
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to latest response
  useEffect(() => {
    if (history.length > 0) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [history, isLoading]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && query.trim() && imagesCount > 0) {
        onSubmit();
      }
    }
  };

  return (
    <div className="flex flex-col space-y-4">
      
      {/* Response History Stream */}
      {history.length > 0 && (
        <div className="space-y-6">
          {history.map((item, idx) => (
            <ResponseCard
              key={idx}
              responseData={item.response}
              queryText={item.query}
              imagePreviews={item.imagePreviews}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>
      )}

      {/* Loading Indicator Card */}
      {isLoading && (
        <div className="bg-slate-900/80 border border-cyan-500/30 rounded-2xl p-6 shadow-xl flex flex-col items-center justify-center text-center space-y-3 animate-pulse">
          <div className="relative flex items-center justify-center">
            <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
            <div className="absolute inset-0 rounded-full border border-cyan-400 animate-ping opacity-30" />
          </div>
          <div>
            <h4 className="text-sm font-semibold text-slate-200">
              Agentic Pipeline Processing
            </h4>
            <p className="text-xs text-cyan-400/80 font-mono mt-0.5">
              Validating tensors • Routing specialist task • Querying Vision-LLM...
            </p>
          </div>
        </div>
      )}

      {/* Query Input Box Card */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl">
        
        {/* Suggested Queries Chips */}
        {suggestedQueries.length > 0 && (
          <div className="mb-3">
            <div className="flex items-center gap-1.5 text-[11px] font-mono text-slate-400 mb-2">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
              <span>Recommended Remote Sensing Inquiries:</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {suggestedQueries.map((sug, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setQuery(sug)}
                  disabled={isLoading}
                  className="text-left text-xs bg-slate-950/70 hover:bg-slate-800 text-slate-300 hover:text-cyan-300 border border-slate-800 hover:border-cyan-500/40 rounded-lg px-2.5 py-1.5 transition-all flex items-center gap-1 group"
                >
                  <span className="line-clamp-1">{sug}</span>
                  <ArrowUpRight className="w-3 h-3 text-slate-500 group-hover:text-cyan-400 flex-shrink-0" />
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input Form */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (!isLoading && query.trim() && imagesCount > 0) {
              onSubmit();
            }
          }}
          className="relative"
        >
          <textarea
            ref={textareaRef}
            rows={3}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              imagesCount === 0
                ? "First upload 1 or 2 satellite images above (or click a Preset)..."
                : imagesCount === 2
                ? "Ask about bi-temporal differences (floods/damage) or Optical+SAR fusion..."
                : "Ask about objects, counts, spatial location (grounding), or scene description..."
            }
            disabled={isLoading || imagesCount === 0}
            className="w-full bg-slate-950/80 border border-slate-800 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 rounded-xl p-3 pr-24 text-sm text-slate-100 placeholder-slate-500 resize-none transition-all disabled:opacity-50 disabled:cursor-not-allowed font-sans leading-relaxed"
          />

          {/* Action Buttons in Bottom Right of Textarea */}
          <div className="absolute right-2.5 bottom-3 flex items-center gap-2">
            
            {/* Effort Control Dropdown */}
            <div className="flex items-center gap-1.5 bg-slate-900/60 border border-slate-700/80 rounded-lg px-2 py-1.5">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Effort:</span>
              <select
                value={effort}
                onChange={(e) => setEffort(e.target.value)}
                disabled={isLoading}
                className="bg-transparent text-xs text-slate-200 font-medium focus:outline-none focus:ring-0 border-none p-0 cursor-pointer disabled:opacity-50"
              >
                <option value="min" className="bg-slate-800 text-slate-200">Min</option>
                <option value="medium" className="bg-slate-800 text-slate-200">Medium</option>
                <option value="max" className="bg-slate-800 text-slate-200">Max</option>
              </select>
            </div>

            {history.length > 0 && (
              <button
                type="button"
                onClick={onClearHistory}
                disabled={isLoading}
                className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-rose-400 border border-slate-700 transition-colors disabled:opacity-50"
                title="Clear conversation history"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )}

            <button
              type="submit"
              disabled={isLoading || !query.trim() || imagesCount === 0}
              className="px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white font-medium text-xs shadow-lg shadow-cyan-500/20 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center gap-1.5"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Analyzing</span>
                </>
              ) : (
                <>
                  <span>Execute</span>
                  <Send className="w-3.5 h-3.5" />
                </>
              )}
            </button>
          </div>
        </form>

        {/* Helper Footer */}
        <div className="mt-2 flex items-center justify-between text-[11px] font-mono text-slate-400">
          <span>Press <kbd className="px-1 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">Enter</kbd> to submit, <kbd className="px-1 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">Shift+Enter</kbd> for newline</span>
          {imagesCount === 0 && (
            <span className="text-amber-400/90 font-medium">⚠️ Satellite image required before executing query</span>
          )}
        </div>

      </div>

    </div>
  );
}
