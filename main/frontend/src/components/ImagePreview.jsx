import { motion } from 'framer-motion';
import { X, HardDrive } from 'lucide-react';
import { IMAGE_TYPE_LABELS } from '../constants';

/**
 * Displays the uploaded image thumbnail with a detected image-type badge.
 *
 * Props:
 *   previewUrl — object URL of the selected image
 *   fileName  — original file name
 *   fileSize  — file size in bytes
 *   imageType — detected type from API: 'xray' | 'mri' | 'skin' | null
 *   onClear() — callback to go back to idle
 *   showClear — whether to show the clear button
 */
const ImagePreview = ({ previewUrl, fileName, fileSize, imageType, onClear, showClear = true }) => {
  if (!previewUrl) return null;

  const typeLabel = imageType ? IMAGE_TYPE_LABELS[imageType] || imageType.toUpperCase() : null;

  const formatSize = (bytes) => {
    if (!bytes) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className="relative rounded-card overflow-hidden border border-slate-700/50 bg-slate-950 group"
    >
      {/* Image */}
      <div className="aspect-square sm:aspect-[4/3] flex items-center justify-center bg-black/40 overflow-hidden">
        <img
          src={previewUrl}
          alt={`Uploaded medical image: ${fileName || 'scan'}`}
          className="w-full h-full object-contain"
        />
      </div>

      {/* Bottom bar with filename + size + type badge */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-slate-900/95 via-slate-900/70 to-transparent p-3 flex items-end justify-between">
        <div className="flex items-center gap-2 min-w-0 max-w-[65%]">
          <span className="text-xs text-slate-400 truncate">{fileName}</span>
          {fileSize > 0 && (
            <span className="flex items-center gap-1 text-[10px] text-slate-500 flex-shrink-0">
              <HardDrive className="w-3 h-3" />
              {formatSize(fileSize)}
            </span>
          )}
        </div>
        {typeLabel && (
          <span className="px-2.5 py-0.5 text-[11px] font-bold rounded-full bg-primary/15 text-primary border border-primary/25 uppercase tracking-wider flex-shrink-0">
            {typeLabel}
          </span>
        )}
      </div>

      {/* Clear button */}
      {showClear && onClear && (
        <button
          onClick={onClear}
          className="absolute top-2 right-2 p-1.5 rounded-full bg-slate-900/70 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors border border-slate-700/50"
          aria-label="Remove image"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </motion.div>
  );
};

export default ImagePreview;
