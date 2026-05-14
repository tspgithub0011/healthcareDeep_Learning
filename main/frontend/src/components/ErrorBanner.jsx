import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertCircle, X, RotateCcw } from 'lucide-react';

/**
 * User-friendly error banner with retry button.
 * Auto-dismisses after 10 seconds unless user interacts.
 *
 * Props:
 *   message  — error text
 *   onRetry  — callback for retry
 *   onDismiss — callback to remove banner
 */
const ErrorBanner = ({ message, onRetry, onDismiss }) => {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(false);
      onDismiss?.();
    }, 10000);

    return () => clearTimeout(timer);
  }, [onDismiss]);

  if (!visible || !message) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -8, height: 0 }}
        animate={{ opacity: 1, y: 0, height: 'auto' }}
        exit={{ opacity: 0, y: -8, height: 0 }}
        className="w-full rounded-card border border-red-500/30 bg-red-500/8 p-4"
      >
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-red-300 leading-relaxed">{message}</p>
          </div>
          <div className="flex items-center gap-1.5 flex-shrink-0">
            {onRetry && (
              <button
                onClick={onRetry}
                className="flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium rounded-btn bg-red-500/15 text-red-300 hover:bg-red-500/25 transition-colors border border-red-500/20"
              >
                <RotateCcw className="w-3 h-3" />
                Retry
              </button>
            )}
            <button
              onClick={() => { setVisible(false); onDismiss?.(); }}
              className="p-1.5 rounded-btn text-red-400/60 hover:text-red-300 transition-colors"
              aria-label="Dismiss error"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default ErrorBanner;
