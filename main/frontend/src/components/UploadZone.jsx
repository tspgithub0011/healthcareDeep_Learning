import { useState, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import { UploadCloud, FileImage, X } from 'lucide-react';
import { ALLOWED_FILE_TYPES, MAX_FILE_SIZE } from '../constants';

/**
 * Drag-and-drop + click-to-browse upload zone.
 * States: idle → file-selected → uploading → done/error
 *
 * Props:
 *   onFileSelect(file) — called when a valid file is chosen
 *   disabled — disables interaction during upload
 */
const UploadZone = ({ onFileSelect, disabled = false }) => {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (disabled) return;

    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) onFileSelect(droppedFile);
  }, [disabled, onFileSelect]);

  const handleInputChange = useCallback((e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) onFileSelect(selectedFile);
    // Reset so the same file can be re-selected
    e.target.value = '';
  }, [onFileSelect]);

  const handleClick = () => {
    if (!disabled) inputRef.current?.click();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="w-full"
    >
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload medical image — drag and drop or click to browse"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        className={`
          relative cursor-pointer rounded-card border-2 border-dashed
          transition-all duration-300 ease-out
          flex flex-col items-center justify-center
          p-10 sm:p-14 text-center
          focus:outline-none focus:ring-2 focus:ring-primary/50 focus:ring-offset-2 focus:ring-offset-slate-900
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          ${isDragging
            ? 'border-primary bg-primary/5 shadow-[0_0_40px_rgba(14,165,233,0.1)]'
            : 'border-slate-700/60 bg-surface/30 hover:border-slate-600 hover:bg-surface/50'
          }
        `}
      >
        {/* Icon */}
        <motion.div
          animate={isDragging ? { y: -6, scale: 1.08 } : { y: 0, scale: 1 }}
          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
          className={`
            p-4 rounded-2xl mb-5 transition-colors duration-300
            ${isDragging
              ? 'bg-primary/10 text-primary'
              : 'bg-slate-800 text-slate-400 border border-slate-700/50'
            }
          `}
        >
          <UploadCloud className="w-10 h-10" />
        </motion.div>

        {/* Text */}
        <h3 className="text-lg font-semibold text-slate-200 mb-1.5">
          {isDragging ? 'Drop your scan here' : 'Drop a medical image here'}
        </h3>
        <p className="text-sm text-slate-400 mb-5 max-w-sm">
          Supports X-Ray, MRI, and Dermatological photos.
          JPEG, PNG, or WEBP — max 10 MB.
        </p>

        {/* Browse button */}
        <span className="inline-flex items-center gap-2 px-5 py-2.5 rounded-btn bg-primary hover:bg-primary/90 text-white text-sm font-semibold shadow-lg shadow-primary/20 transition-all duration-200 active:scale-95">
          <FileImage className="w-4 h-4" />
          Browse Files
        </span>

        {/* Hidden input */}
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept={ALLOWED_FILE_TYPES.join(',')}
          onChange={handleInputChange}
          disabled={disabled}
        />
      </div>
    </motion.div>
  );
};

export default UploadZone;
