import { AlertTriangle } from 'lucide-react';

/**
 * Persistent medical disclaimer banner — cannot be dismissed.
 * Stays at the very top of the page at all times.
 */
const MedicalDisclaimer = () => {
  return (
    <div className="w-full bg-amber-500/10 border-b border-amber-500/20 py-2.5 px-4 flex-shrink-0">
      <div className="max-w-7xl mx-auto flex items-center justify-center gap-2.5 text-center">
        <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
        <p className="text-xs sm:text-sm text-amber-300/90 font-medium leading-snug">
          <span className="font-semibold text-amber-200">Disclaimer:</span>{' '}
          This tool is for <span className="font-semibold">educational purposes only</span>. 
          It does not provide medical diagnoses. Always consult a qualified healthcare professional.
        </p>
      </div>
    </div>
  );
};

export default MedicalDisclaimer;
