import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import PresetBar from './components/PresetBar';
import ImageUpload from './components/ImageUpload';
import ChatWindow from './components/ChatWindow';
import ArchitectureModal from './components/ArchitectureModal';
import { AlertCircle, RefreshCw, Satellite, Radio, Info } from 'lucide-react';

const FALLBACK_PRESETS = [
  {
    id: "flood-change",
    title: "🌊 Urban Flood Inundation Analysis",
    category: "change_detection",
    tag: "Bi-Temporal CDVQA (2 Images)",
    description: "Compare pre-event and post-event satellite imagery to map inundated streets and canal overflow.",
    image_names: ["urban_before.jpg", "urban_after_flood.jpg"],
    suggested_query: "Compare these before and after satellite images and identify all flood-damaged areas and inundated roads."
  },
  {
    id: "sar-fusion",
    title: "🛰️ Optical + SAR Crop Vitality Fusion",
    category: "optical_sar_fusion",
    tag: "Multi-Sensor Fusion (2 Images)",
    description: "Fuse visible optical RGB reflectance with Sentinel-1 SAR microwave radar backscatter for agricultural moisture mapping.",
    image_names: ["agri_optical.jpg", "agri_sar.jpg"],
    suggested_query: "Analyze the multi-modal optical and SAR radar imagery to determine surface roughness, soil moisture, and crop health."
  },
  {
    id: "port-grounding",
    title: "🎯 Naval Port & Storage Grounding",
    category: "grounding",
    tag: "Visual Grounding (1 Image)",
    description: "Detect and spatially localize fuel oil storage tanks, container ships, and dockside gantry cranes with coordinates.",
    image_names: ["naval_port_grounding.jpg"],
    suggested_query: "Where are the fuel oil storage tanks, container ships, and gantry cranes located? Provide spatial coordinates and quadrants."
  },
  {
    id: "airport-captioning",
    title: "📝 International Airport Scene Captioning",
    category: "captioning",
    tag: "Scene Summary (1 Image)",
    description: "Generate a multi-scale land-use land-cover (LULC) scene taxonomy, transit corridors, and aviation infrastructure summary.",
    image_names: ["airport_hub_captioning.jpg"],
    suggested_query: "Describe this satellite scene in detail, including primary land cover taxonomy, transit corridors, and aviation infrastructure."
  },
  {
    id: "solar-vqa",
    title: "🔍 Solar Park & Turbine Counting",
    category: "single_image_vqa",
    tag: "Single-Image RSVQA (1 Image)",
    description: "Perform precision visual question answering for solar array grid structure, substation location, and wind turbine counts.",
    image_names: ["solar_park_vqa.jpg"],
    suggested_query: "How many wind turbines are visible on the western perimeter, and what is the layout of the solar photovoltaic array and substation?"
  }
];

export default function App() {
  const [images, setImages] = useState([]);
  const [query, setQuery] = useState('');
  const [effort, setEffort] = useState('medium');
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [presets, setPresets] = useState(FALLBACK_PRESETS);
  const [activePresetId, setActivePresetId] = useState(null);
  const [backendHealth, setBackendHealth] = useState(null);
  const [isArchitectureOpen, setIsArchitectureOpen] = useState(false);

  // Fetch backend health and sample presets on initial mount
  useEffect(() => {
    checkHealth();
    fetchPresets();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setBackendHealth(data);
      } else {
        setBackendHealth({ status: 'error', active_model: 'Backend Unreachable' });
      }
    } catch (e) {
      setBackendHealth({ status: 'error', active_model: 'Backend Offline' });
    }
  };

  const fetchPresets = async () => {
    try {
      const res = await fetch('/api/samples');
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) {
          setPresets(data);
        }
      }
    } catch (e) {
      // Use fallback presets if fetch fails
      setPresets(FALLBACK_PRESETS);
    }
  };

  // Handle Preset Selection
  const handleSelectPreset = async (preset) => {
    setActivePresetId(preset.id);
    setQuery(preset.suggested_query);
    setError(null);

    try {
      // Load preset images as File objects with instant previewUrl
      const loadedFiles = await Promise.all(
        preset.image_names.map(async (name) => {
          const res = await fetch(`/sample-images/${name}`);
          const blob = await res.blob();
          const file = new File([blob], name, { type: blob.type || 'image/jpeg' });
          file.previewUrl = `/sample-images/${name}`;
          return file;
        })
      );
      setImages(loadedFiles);
    } catch (e) {
      console.error("Failed to load preset images via fetch:", e);
      // Fallback with virtual image objects
      setImages(
        preset.image_names.map((name) => ({
          name,
          url: `/sample-images/${name}`,
          previewUrl: `/sample-images/${name}`
        }))
      );
    }
  };

  // Remove individual image
  const handleRemoveImage = (index) => {
    const next = [...images];
    next.splice(index, 1);
    setImages(next);
    setActivePresetId(null);
  };

  // Submit Query to Backend API
  const handleSubmitQuery = async () => {
    if (images.length === 0) {
      setError("Please upload at least 1 satellite image before running an inquiry.");
      return;
    }
    if (!query.trim()) {
      setError("Please enter a question or instruction regarding the satellite imagery.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('query', query.trim());
      formData.append('effort', effort);

      for (let i = 0; i < images.length; i++) {
        const imgObj = images[i];
        if (imgObj instanceof File) {
          formData.append('images', imgObj);
        } else if (imgObj.url) {
          // Fetch the blob and attach
          const res = await fetch(imgObj.url);
          const blob = await res.blob();
          const file = new File([blob], imgObj.name || `image_${i + 1}.jpg`, { type: blob.type || 'image/jpeg' });
          formData.append('images', file);
        }
      }

      const res = await fetch('/api/query', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Server returned error ${res.status}: ${res.statusText}`);
      }

      const responseData = await res.json();

      // Collect image preview URLs for the history record
      const previews = images.map(img => img instanceof File ? URL.createObjectURL(img) : img.url);

      setHistory(prev => [
        ...prev,
        {
          query: query.trim(),
          response: responseData,
          imagePreviews: previews
        }
      ]);

      // Clear query input for next turn
      setQuery('');

    } catch (err) {
      console.error("Query execution error:", err);
      setError(err.message || "An unexpected error occurred while communicating with the SatQuery AI engine.");
    } finally {
      setIsLoading(false);
    }
  };

  // Determine dynamic suggested queries based on active image count
  const getSuggestedQueries = () => {
    if (images.length === 2) {
      return [
        "Compare these before and after satellite images and identify all flood-damaged areas.",
        "Analyze the multi-modal optical and SAR radar imagery to determine surface roughness and soil moisture.",
        "Detect any newly constructed buildings or land-use changes between these two temporal timestamps."
      ];
    } else if (images.length === 1) {
      return [
        "Where are the cargo vessels and gantry cranes located? Provide bounding box coordinates.",
        "Describe this satellite scene in detail, including primary land cover taxonomy and infrastructure.",
        "How many large container cargo vessels are berthed at the docks?"
      ];
    }
    return [];
  };

  return (
    <div className="min-h-screen bg-space-950 text-slate-100 bg-grid-pattern relative flex flex-col">
      
      {/* Top Header */}
      <Header
        backendHealth={backendHealth}
        onOpenArchitecture={() => setIsArchitectureOpen(true)}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        
        {/* Preset Bar */}
        <PresetBar
          presets={presets}
          activePresetId={activePresetId}
          onSelectPreset={handleSelectPreset}
        />

        {/* Error Alert Banner */}
        {error && (
          <div className="p-4 rounded-xl bg-rose-950/70 border border-rose-700/80 text-rose-200 flex items-start justify-between gap-3 shadow-lg">
            <div className="flex items-start gap-2.5">
              <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider font-mono text-rose-300">
                  Execution Notice
                </h4>
                <p className="text-xs text-rose-100 mt-0.5 leading-relaxed">
                  {error}
                </p>
              </div>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-xs font-mono text-rose-300 hover:text-white px-2 py-1 bg-rose-900/60 rounded-md border border-rose-700"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Workspace Layout Grid */}
        <div className="space-y-6">
          
          {/* Image Uploader & Dual-Tensor Inspector */}
          <ImageUpload
            images={images}
            onImagesChange={setImages}
            onRemoveImage={handleRemoveImage}
            disabled={isLoading}
          />

          {/* Query Input Box & Response Stream */}
          <ChatWindow
            history={history}
            query={query}
            setQuery={setQuery}
            effort={effort}
            setEffort={setEffort}
            onSubmit={handleSubmitQuery}
            isLoading={isLoading}
            onClearHistory={() => setHistory([])}
            imagesCount={images.length}
            suggestedQueries={getSuggestedQueries()}
          />

        </div>

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-space-900/40 py-4 text-center text-xs font-mono text-slate-400">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400" />
            <span>SatQuery AI — ISRO Remote Sensing Vision-Language Assistant Prototype</span>
          </div>
          <div>
            <span>Smart India Hackathon (SIH 2026) • Team Demonstration Build</span>
          </div>
        </div>
      </footer>

      {/* Architecture & Roadmap Modal */}
      <ArchitectureModal
        isOpen={isArchitectureOpen}
        onClose={() => setIsArchitectureOpen(false)}
      />

    </div>
  );
}
