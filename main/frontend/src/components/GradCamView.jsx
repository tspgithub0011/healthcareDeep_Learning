import { useState } from 'react';
import { motion } from 'framer-motion';
import { Eye, Layers, SlidersHorizontal } from 'lucide-react';

/**
 * Grad-CAM heatmap viewer.
 * Supports side-by-side vs overlay toggle + opacity slider.
 *
 * Props:
 *   originalUrl  — preview URL of the original image
 *   heatmapSrc   — base64 data URI of the Grad-CAM heatmap image
 *   modelUsed    — name of the model that generated the heatmap
 */
const GradCamView = ({ originalUrl, heatmapSrc, modelUsed }) => {
  const [mode, setMode] = useState('overlay'); // 'overlay' | 'side-by-side'
  const [opacity, setOpacity] = useState(0.5);

  if (!heatmapSrc) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      className="space-y-3"
    >
      {/* Header row */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
          <Eye className="w-4 h-4 text-secondary" />
          Grad-CAM Heatmap
        </h3>

        {/* Toggle */}
        <div className="flex items-center gap-1 p-0.5 rounded-lg bg-slate-800 border border-slate-700/50">
          <button
            onClick={() => setMode('overlay')}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
              mode === 'overlay'
                ? 'bg-primary/20 text-primary'
                : 'text-slate-400 hover:text-slate-300'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setMode('side-by-side')}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
              mode === 'side-by-side'
                ? 'bg-primary/20 text-primary'
                : 'text-slate-400 hover:text-slate-300'
            }`}
          >
            Side
          </button>
        </div>
      </div>

      {/* Image view */}
      {mode === 'overlay' ? (
        <div className="relative rounded-card overflow-hidden bg-black border border-slate-700/40">
          <img
            src={originalUrl}
            alt="Original medical scan"
            className="w-full h-auto object-contain"
          />
          <img
            src={heatmapSrc}
            alt="Grad-CAM heatmap overlay"
            className="absolute inset-0 w-full h-full object-contain mix-blend-screen"
            style={{ opacity }}
          />
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-card overflow-hidden bg-black border border-slate-700/40">
            <img src={originalUrl} alt="Original scan" className="w-full h-auto object-contain" />
            <p className="text-[10px] text-slate-500 text-center py-1">Original</p>
          </div>
          <div className="rounded-card overflow-hidden bg-black border border-slate-700/40">
            <img src={heatmapSrc} alt="Grad-CAM heatmap" className="w-full h-auto object-contain" />
            <p className="text-[10px] text-slate-500 text-center py-1">Heatmap</p>
          </div>
        </div>
      )}

      {/* Opacity slider (overlay mode only) */}
      {mode === 'overlay' && (
        <div className="flex items-center gap-3 px-1">
          <SlidersHorizontal className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="flex-1 h-1 rounded-full appearance-none bg-slate-700 cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 
              [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:shadow-lg"
            aria-label="Heatmap opacity"
          />
          <span className="text-xs text-slate-500 w-8 text-right">{Math.round(opacity * 100)}%</span>
        </div>
      )}

      {/* Caption */}
      <p className="text-xs text-slate-500 leading-relaxed">
        The highlighted regions show areas the <span className="text-slate-400 font-medium">{modelUsed || 'model'}</span> focused 
        on when making its prediction. Warmer colors indicate higher importance.
      </p>
    </motion.div>
  );
};

export default GradCamView;
