import React, { useRef, useState } from 'react';
import { Expand, Image as ImageIcon, Plus, Upload, X } from 'lucide-react';

export default function ImageUpload({ images, onImagesChange, onRemoveImage, disabled }) {
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [previewModalUrl, setPreviewModalUrl] = useState(null);

  const handleFiles = (newFiles) => {
    const validFiles = newFiles.filter((file) =>
      file.type.startsWith('image/') || /\.(png|jpe?g|webp|bmp)$/i.test(file.name),
    );
    if (!validFiles.length) {
      window.alert('Please upload a PNG, JPG, JPEG, or WebP image.');
      return;
    }
    const prepared = validFiles.slice(0, 2 - images.length).map((file) => {
      file.previewUrl = URL.createObjectURL(file);
      return file;
    });
    if (images.length + validFiles.length > 2) {
      window.alert('SatQuery supports a maximum of two satellite images.');
    }
    onImagesChange([...images, ...prepared]);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragActive(false);
    if (!disabled) handleFiles(Array.from(event.dataTransfer.files || []));
  };

  const handleInput = (event) => {
    handleFiles(Array.from(event.target.files || []));
    event.target.value = '';
  };

  return (
    <section className="surface-panel">
      <div className="panel-header">
        <div className="panel-title">
          <span className="panel-index">01</span>
          <div>
            <h3>Load your imagery</h3>
            <p>Upload a scene or pair two moments / sensors.</p>
          </div>
        </div>
        <div>
          <span className={`image-count ${images.length ? '' : 'empty'}`}>
            {images.length ? `${images.length} / 2 loaded` : 'No image loaded'}
          </span>
          {images.length > 0 && (
            <button type="button" className="clear-button" onClick={() => onImagesChange([])} disabled={disabled}>
              Clear all
            </button>
          )}
        </div>
      </div>

      <div className="image-grid">
        {images.map((image, index) => {
          const name = image instanceof File ? image.name : image.name || `Image ${index + 1}`;
          const previewUrl = image.previewUrl || image.url;
          return (
            <div className="image-tile" key={`${name}-${index}`}>
              <img src={previewUrl} alt={name} />
              <span className="image-label">{index === 0 ? 'Image 01 / source' : 'Image 02 / compare'}</span>
              <div className="image-actions">
                <button type="button" onClick={() => setPreviewModalUrl(previewUrl)} aria-label={`Expand ${name}`}><Expand size={14} /></button>
                {!disabled && <button type="button" onClick={() => onRemoveImage(index)} aria-label={`Remove ${name}`}><X size={14} /></button>}
              </div>
              <div className="image-meta">
                <span>{name}</span>
                <span>{image instanceof File && image.size ? `${(image.size / 1048576).toFixed(2)} MB` : 'Preset scene'}</span>
              </div>
            </div>
          );
        })}

        {images.length < 2 && (
          <div
            className={`upload-zone ${dragActive ? 'is-dragging' : ''}`}
            onDragEnter={(event) => { event.preventDefault(); setDragActive(true); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            onClick={() => !disabled && fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => { if (event.key === 'Enter') fileInputRef.current?.click(); }}
          >
            <input ref={fileInputRef} type="file" accept="image/png,image/jpeg,image/webp" multiple onChange={handleInput} disabled={disabled} />
            <Upload className="upload-icon" size={25} />
            <strong>{images.length ? 'Add a second image' : 'Drop imagery here'}</strong>
            <p>{images.length ? 'Pair a before / after scene or add an Optical + SAR sensor.' : 'PNG, JPG, or WebP · up to two images per inquiry'}</p>
          </div>
        )}
      </div>

      {previewModalUrl && (
        <div className="modal-backdrop" onClick={() => setPreviewModalUrl(null)}>
          <div className="modal-panel" onClick={(event) => event.stopPropagation()} style={{ maxWidth: 1000, padding: 10 }}>
            <button type="button" className="modal-close" onClick={() => setPreviewModalUrl(null)} aria-label="Close image preview"><X size={16} /></button>
            <img src={previewModalUrl} alt="Expanded satellite scene" style={{ width: '100%', maxHeight: '82vh', objectFit: 'contain' }} />
          </div>
        </div>
      )}
    </section>
  );
}