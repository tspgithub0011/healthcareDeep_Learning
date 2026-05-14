import { Activity } from 'lucide-react';

/**
 * Animated heartbeat loading spinner shown during analysis.
 */
const LoadingSpinner = () => {
  return (
    <div className="flex flex-col items-center justify-center py-12 gap-6">
      {/* Pulsing ring + heartbeat icon */}
      <div className="relative">
        <div className="absolute inset-0 w-20 h-20 rounded-full border-2 border-primary/30 animate-pulse-ring -m-2" />
        <div className="w-16 h-16 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center">
          <Activity className="w-8 h-8 text-primary animate-heartbeat drop-shadow-[0_0_8px_rgba(14,165,233,0.5)]" />
        </div>
      </div>

      {/* Text */}
      <div className="text-center">
        <h3 className="text-lg font-semibold text-slate-200 mb-1">Analyzing your image…</h3>
        <p className="text-sm text-slate-400">
          Running deep learning models. This may take a few seconds.
        </p>
      </div>

      {/* Progress bar (indeterminate) */}
      <div className="w-48 h-1 bg-slate-800 rounded-full overflow-hidden">
        <div className="h-full bg-gradient-to-r from-primary to-secondary rounded-full animate-[slideRight_1.5s_ease-in-out_infinite] w-1/3"
          style={{
            animation: 'slideRight 1.5s ease-in-out infinite',
          }}
        />
      </div>

      <style>{`
        @keyframes slideRight {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(400%); }
        }
      `}</style>
    </div>
  );
};

export default LoadingSpinner;
