import React, { useEffect, useState } from 'react';
import Header from './components/Header';
import PresetBar from './components/PresetBar';
import ImageUpload from './components/ImageUpload';
import ChatWindow from './components/ChatWindow';
import ArchitectureModal from './components/ArchitectureModal';
import { AlertCircle, ArrowUpRight, CheckCircle2, Orbit, X } from 'lucide-react';

const FALLBACK_PRESETS = [
  {
    id: 'flood-change',
    title: 'Urban Flood Inundation Analysis',
    category: 'change_detection',
    tag: 'Bi-Temporal CDVQA',
    description: 'Compare pre-event and post-event satellite imagery to map inundated streets and canal overflow.',
    image_names: ['urban_before.jpg', 'urban_after_flood.jpg'],
    suggested_query: 'Compare these before and after satellite images and identify all flood-damaged areas and inundated roads.',
  },
  {
    id: 'sar-fusion',
    title: 'Optical + SAR Crop Vitality Fusion',
    category: 'optical_sar_fusion',
    tag: 'Multi-Sensor Fusion',
    description: 'Fuse visible optical reflectance with Sentinel-1 radar backscatter for agricultural moisture mapping.',
    image_names: ['agri_optical.jpg', 'agri_sar.jpg'],
    suggested_query: 'Analyze the multi-modal optical and SAR radar imagery to determine surface roughness, soil moisture, and crop health.',
  },
  {
    id: 'port-grounding',
    title: 'Naval Port & Storage Grounding',
    category: 'grounding',
    tag: 'Visual Grounding',
    description: 'Detect and spatially localize fuel oil storage tanks, container ships, and dockside gantry cranes.',
    image_names: ['naval_port_grounding.jpg'],
    suggested_query: 'Where are the fuel oil storage tanks, container ships, and gantry cranes located? Provide spatial coordinates and quadrants.',
  },
  {
    id: 'airport-captioning',
    title: 'International Airport Scene Captioning',
    category: 'captioning',
    tag: 'Scene Summary',
    description: 'Generate a land-use taxonomy, transit corridor overview, and aviation infrastructure summary.',
    image_names: ['airport_hub_captioning.jpg'],
    suggested_query: 'Describe this satellite scene in detail, including primary land cover taxonomy, transit corridors, and aviation infrastructure.',
  },
  {
    id: 'solar-vqa',
    title: 'Solar Park & Turbine Counting',
    category: 'single_image_vqa',
    tag: 'Single-Image VQA',
    description: 'Answer precise questions about solar array structure, substations, and visible wind turbine counts.',
    image_names: ['solar_park_vqa.jpg'],
    suggested_query: 'How many wind turbines are visible on the western perimeter, and what is the layout of the solar photovoltaic array and substation?',
  },
];

const pipeline = [
  ['01', 'Ingest', 'Image + intent'],
  ['02', 'Route', 'Specialist task'],
  ['03', 'Infer', 'Vision analysis'],
  ['04', 'Explain', 'Evidence + trace'],
];

export default function App() {
  const [images, setImages] = useState([]);
  const [query, setQuery] = useState('');
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [presets, setPresets] = useState(FALLBACK_PRESETS);
  const [activePresetId, setActivePresetId] = useState(null);
  const [backendHealth, setBackendHealth] = useState(null);
  const [isArchitectureOpen, setIsArchitectureOpen] = useState(false);

  useEffect(() => {
    checkHealth();
    fetchPresets();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('Backend unavailable');
      setBackendHealth(await res.json());
    } catch {
      setBackendHealth({ status: 'error', active_model: 'Backend offline' });
    }
  };

  const fetchPresets = async () => {
    try {
      const res = await fetch('/api/samples');
      if (!res.ok) throw new Error('Preset request failed');
      const data = await res.json();
      if (Array.isArray(data) && data.length) setPresets(data);
    } catch {
      setPresets(FALLBACK_PRESETS);
    }
  };

  const handleSelectPreset = async (preset) => {
    setActivePresetId(preset.id);
    setQuery(preset.suggested_query);
    setError(null);

    try {
      const loadedFiles = await Promise.all(
        preset.image_names.map(async (name) => {
          const res = await fetch(`/sample-images/${name}`);
          if (!res.ok) throw new Error(`Could not load ${name}`);
          const blob = await res.blob();
          const file = new File([blob], name, { type: blob.type || 'image/jpeg' });
          file.previewUrl = `/sample-images/${name}`;
          return file;
        }),
      );
      setImages(loadedFiles);
    } catch (e) {
      console.error('Failed to load preset images:', e);
      setImages(preset.image_names.map((name) => ({
        name,
        url: `/sample-images/${name}`,
        previewUrl: `/sample-images/${name}`,
      })));
    }
  };

  const handleRemoveImage = (index) => {
    setImages((current) => current.filter((_, imageIndex) => imageIndex !== index));
    setActivePresetId(null);
  };

  const handleSubmitQuery = async () => {
    if (!images.length) {
      setError('Add at least one satellite image before running an inquiry.');
      return;
    }
    if (!query.trim()) {
      setError('Enter a question or instruction about the satellite imagery.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('query', query.trim());

      for (let index = 0; index < images.length; index += 1) {
        const image = images[index];
        if (image instanceof File) {
          formData.append('images', image);
        } else if (image.url) {
          const res = await fetch(image.url);
          if (!res.ok) throw new Error(`Could not attach ${image.name || 'image'}`);
          const blob = await res.blob();
          formData.append('images', new File([blob], image.name || `image_${index + 1}.jpg`, {
            type: blob.type || 'image/jpeg',
          }));
        }
      }

      const res = await fetch('/api/query', { method: 'POST', body: formData });
      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Server returned error ${res.status}`);
      }

      const responseData = await res.json();
      setHistory((current) => [
        ...current,
        {
          query: query.trim(),
          response: responseData,
          imagePreviews: images.map((image) => image.previewUrl || image.url || ''),
        },
      ]);
      setQuery('');
    } catch (err) {
      console.error('Query execution error:', err);
      setError(err.message || 'Something went wrong while contacting the SatQuery engine.');
    } finally {
      setIsLoading(false);
    }
  };

  const getSuggestedQueries = () => {
    if (images.length === 2) {
      return [
        'Compare these before and after images and identify flood-damaged areas.',
        'Analyze optical and SAR imagery for surface roughness and soil moisture.',
        'Detect newly constructed buildings or land-use changes between these timestamps.',
      ];
    }
    if (images.length === 1) {
      return [
        'Where are the cargo vessels and gantry cranes located?',
        'Describe this scene, including land cover and infrastructure.',
        'How many large container cargo vessels are berthed at the docks?',
      ];
    }
    return [];
  };

  const isHealthy = backendHealth?.status === 'ok';

  return (
    <div className="app-shell">
      <Header
        backendHealth={backendHealth}
        onOpenArchitecture={() => setIsArchitectureOpen(true)}
      />

      <main className="app-main">
        <section className="intro-grid">
          <div className="intro-copy">
            <div className="eyebrow"><span className="eyebrow-mark" /> Remote sensing intelligence</div>
            <h2>Turn satellite scenes into <em>clear decisions.</em></h2>
            <p>
              Ask natural language questions across optical, SAR, and bi-temporal imagery.
              SatQuery routes every inquiry to the right specialist and shows its work.
            </p>
          </div>
          <div className="intro-note">
            <Orbit className="intro-orbit" size={18} />
            <span>Built for earth observation teams</span>
            <strong>One question. Every signal.</strong>
          </div>
        </section>

        <section className="preset-section" aria-labelledby="presets-heading">
          <div className="section-heading">
            <div>
              <span className="section-kicker">Start with a signal</span>
              <h3 id="presets-heading">Field presets</h3>
            </div>
            <span className="section-meta">Five ready-to-run investigations</span>
          </div>
          <PresetBar
            presets={presets}
            activePresetId={activePresetId}
            onSelectPreset={handleSelectPreset}
          />
        </section>

        {error && (
          <div className="notice notice-error" role="alert">
            <div className="notice-icon"><AlertCircle size={17} /></div>
            <div>
              <strong>Execution notice</strong>
              <p>{error}</p>
            </div>
            <button type="button" className="icon-button" onClick={() => setError(null)} aria-label="Dismiss notice">
              <X size={16} />
            </button>
          </div>
        )}

        <section className="workspace-grid" aria-label="Satellite analysis workspace">
          <div className="workspace-primary">
            <ImageUpload
              images={images}
              onImagesChange={setImages}
              onRemoveImage={handleRemoveImage}
              disabled={isLoading}
            />
            <ChatWindow
              history={history}
              query={query}
              setQuery={setQuery}
              onSubmit={handleSubmitQuery}
              isLoading={isLoading}
              onClearHistory={() => setHistory([])}
              imagesCount={images.length}
              suggestedQueries={getSuggestedQueries()}
            />
          </div>

          <aside className="workspace-aside">
            <div className="aside-card aside-status">
              <div className="aside-card-heading">
                <span className="section-kicker">Live system</span>
                <span className={`status-dot ${isHealthy ? 'is-on' : ''}`} />
              </div>
              <strong>{isHealthy ? 'Ready for analysis' : 'Connecting to engine'}</strong>
              <p>{isHealthy ? 'Your workspace is connected to the SatQuery backend.' : 'Start the backend service to enable live analysis.'}</p>
              <div className="status-line">
                <span>Model</span>
                <span title={backendHealth?.active_model}>{backendHealth?.active_model || 'Waiting…'}</span>
              </div>
            </div>

            <div className="aside-card">
              <div className="aside-card-heading">
                <span className="section-kicker">The pipeline</span>
                <CheckCircle2 size={16} className="aside-check" />
              </div>
              <div className="pipeline-list">
                {pipeline.map(([number, title, detail]) => (
                  <div className="pipeline-step" key={number}>
                    <span className="pipeline-number">{number}</span>
                    <div><strong>{title}</strong><span>{detail}</span></div>
                    <ArrowUpRight size={14} />
                  </div>
                ))}
              </div>
              <button type="button" className="text-button" onClick={() => setIsArchitectureOpen(true)}>
                View the full architecture <ArrowUpRight size={14} />
              </button>
            </div>

            <div className="aside-quote">
              <span>“</span>
              <p>Good intelligence makes the complex feel legible.</p>
              <small>SatQuery field note / 01</small>
            </div>
          </aside>
        </section>
      </main>

      <footer className="app-footer">
        <span><span className="footer-mark" /> SatQuery AI</span>
        <span>Agentic vision-language analysis for earth observation</span>
        <span>SIH 2026 / ISRO</span>
      </footer>

      <ArchitectureModal
        isOpen={isArchitectureOpen}
        onClose={() => setIsArchitectureOpen(false)}
      />
    </div>
  );
}