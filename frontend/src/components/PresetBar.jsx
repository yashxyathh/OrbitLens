import React from 'react';
import { ArrowUpRight } from 'lucide-react';

const imagesFor = (preset) => preset.image_names?.[0] || 'urban_before.jpg';

export default function PresetBar({ presets, activePresetId, onSelectPreset }) {
  if (!presets?.length) return null;

  return (
    <div className="preset-list">
      {presets.map((preset) => {
        const isSelected = activePresetId === preset.id;
        const title = (preset.title || '').replace(/^[^\w]+/, '');
        return (
          <button
            type="button"
            key={preset.id}
            onClick={() => onSelectPreset(preset)}
            className={`preset-card ${isSelected ? 'is-selected' : ''}`}
          >
            <img className="preset-image" src={`/sample-images/${imagesFor(preset)}`} alt="" />
            <div className="preset-top">
              <span className="preset-tag">{preset.tag || preset.category?.replaceAll('_', ' ')}</span>
              <span className="preset-count">{preset.image_names?.length || 0} {preset.image_names?.length === 1 ? 'image' : 'images'}</span>
            </div>
            <h4>{title}</h4>
            <p>{preset.description}</p>
            <ArrowUpRight className="preset-arrow" size={16} />
          </button>
        );
      })}
    </div>
  );
}