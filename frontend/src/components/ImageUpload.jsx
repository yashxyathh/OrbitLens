import React, { useRef, useState } from 'react';
import { Upload, X, Eye, Image as ImageIcon, Plus, Maximize2, AlertCircle } from 'lucide-react';

export default function ImageUpload({ images, onImagesChange, onRemoveImage, disabled }) {
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [previewModalUrl, setPreviewModalUrl] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(Array.from(e.target.files));
    }
    // reset input so re-selecting same file fires change event
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleFiles = (newFiles) => {
    const validImageFiles = newFiles.filter(file => 
      file.type.startsWith('image/') || /\.(png|jpe?g|webp|bmp)$/i.test(file.name)
    );

    if (validImageFiles.length === 0) {
      alert("Please upload standard image formats (PNG, JPG, JPEG, WebP).");
      return;
    }

    const combined = [...images, ...validImageFiles].slice(0, 2);
    if (images.length + validImageFiles.length > 2) {
      alert("Maximum 2 satellite images allowed (Single Image or Dual-Sensor/Bi-Temporal Pair).");
    }
    onImagesChange(combined);
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 shadow-xl">
      
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <ImageIcon className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-slate-200">
            Satellite Imagery Tensor Input
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-xs font-mono px-2.5 py-0.5 rounded-full border ${
            images.length === 0 ? 'bg-slate-800 text-slate-400 border-slate-700' :
            images.length === 1 ? 'bg-cyan-950 text-cyan-300 border-cyan-800' :
            'bg-indigo-950 text-indigo-300 border-indigo-800'
          }`}>
            {images.length}/2 {images.length === 2 ? 'Dual-Image Pair' : images.length === 1 ? 'Single Image' : 'No Image'}
          </span>
          {images.length > 0 && (
            <button
              onClick={() => onImagesChange([])}
              disabled={disabled}
              className="text-xs text-rose-400 hover:text-rose-300 font-mono hover:underline disabled:opacity-50"
            >
              Clear All
            </button>
          )}
        </div>
      </div>

      {/* Upload Zone / Dual Preview Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        
        {/* Render Uploaded Images */}
        {images.map((imgObj, index) => {
          const isSecond = index === 1;
          const fileName = imgObj instanceof File ? imgObj.name : imgObj.name || `Image_${index+1}.jpg`;
          const previewUrl = imgObj.previewUrl || (imgObj instanceof File ? URL.createObjectURL(imgObj) : imgObj.url);
          const fileSize = imgObj instanceof File && imgObj.size ? `${(imgObj.size / (1024 * 1024)).toFixed(2)} MB` : 'Satellite Tensor';

          return (
            <div
              key={`${fileName}_${index}`}
              className="relative group bg-slate-950/80 border border-slate-700/80 hover:border-cyan-500/50 rounded-xl overflow-hidden transition-all flex flex-col"
            >
              {/* Image Preview Window */}
              <div className="relative aspect-video w-full bg-black/50 overflow-hidden flex items-center justify-center">
                <img
                  key={`${fileName}_img_${index}`}
                  src={previewUrl}
                  alt={fileName}
                  className="w-full h-full object-cover object-center group-hover:scale-105 transition-transform duration-300"
                />

                {/* Overlay Badge */}
                <div className="absolute top-2 left-2 flex items-center gap-1.5 bg-slate-950/80 backdrop-blur-md border border-slate-700 px-2 py-0.5 rounded-md text-[11px] font-mono">
                  <span className={`w-2 h-2 rounded-full ${isSecond ? 'bg-indigo-400' : 'bg-cyan-400'}`} />
                  <span className="text-slate-200 font-medium">
                    {isSecond ? 'Image 2 (T1 / SAR / After)' : 'Image 1 (T0 / Optical / Before)'}
                  </span>
                </div>

                {/* Quick Actions Hover */}
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                  <button
                    onClick={() => setPreviewModalUrl(previewUrl)}
                    className="p-2 rounded-lg bg-slate-900/90 text-slate-200 hover:text-white hover:bg-slate-800 border border-slate-700 shadow-lg"
                    title="Zoom Full Resolution"
                  >
                    <Maximize2 className="w-4 h-4" />
                  </button>
                  {!disabled && (
                    <button
                      onClick={() => onRemoveImage(index)}
                      className="p-2 rounded-lg bg-rose-950/90 text-rose-300 hover:text-rose-100 hover:bg-rose-900 border border-rose-800 shadow-lg"
                      title="Remove Image"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>

              {/* Metadata Footer */}
              <div className="p-2.5 bg-slate-900/60 border-t border-slate-800 flex items-center justify-between text-[11px] font-mono text-slate-400">
                <span className="truncate max-w-[180px] text-slate-300 font-medium" title={fileName}>
                  {fileName}
                </span>
                <span>{fileSize}</span>
              </div>
            </div>
          );
        })}

        {/* Upload Dropzone (if fewer than 2 images uploaded) */}
        {images.length < 2 && (
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-4 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 min-h-[140px] ${
              dragActive
                ? 'border-cyan-400 bg-cyan-950/30'
                : 'border-slate-800 hover:border-slate-700 bg-slate-950/40 hover:bg-slate-950/70'
            } ${images.length === 0 ? 'md:col-span-2' : ''}`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/jpg,image/webp"
              multiple
              onChange={handleFileInput}
              disabled={disabled}
              className="hidden"
            />
            <div className="w-10 h-10 rounded-full bg-slate-800/80 border border-slate-700 flex items-center justify-center mb-2 text-cyan-400 group-hover:scale-110 transition-transform">
              {images.length === 0 ? <Upload className="w-5 h-5" /> : <Plus className="w-5 h-5" />}
            </div>
            <p className="text-xs font-semibold text-slate-200">
              {images.length === 0 ? 'Drop 1 or 2 Satellite Images here' : 'Add 2nd Satellite Image (Pair / SAR)'}
            </p>
            <p className="text-[11px] text-slate-400 mt-1">
              Supports PNG, JPG, WebP • Optical, SAR, or Bi-Temporal pairs
            </p>
          </div>
        )}

      </div>

      {/* Fullscreen Zoom Modal */}
      {previewModalUrl && (
        <div
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4"
          onClick={() => setPreviewModalUrl(null)}
        >
          <div className="relative max-w-4xl max-h-[90vh] bg-slate-900 border border-slate-700 rounded-2xl overflow-hidden shadow-2xl p-2">
            <button
              onClick={() => setPreviewModalUrl(null)}
              className="absolute top-4 right-4 p-2 rounded-full bg-slate-950/80 text-slate-300 hover:text-white border border-slate-700 z-10"
            >
              <X className="w-5 h-5" />
            </button>
            <img
              src={previewModalUrl}
              alt="High Resolution Satellite View"
              className="max-w-full max-h-[85vh] object-contain rounded-xl"
            />
          </div>
        </div>
      )}

    </div>
  );
}
