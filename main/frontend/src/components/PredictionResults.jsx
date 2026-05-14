import { motion } from 'framer-motion';
import { ShieldAlert, ShieldCheck, ShieldQuestion, Trophy } from 'lucide-react';
import { RISK_THRESHOLDS } from '../constants';

/**
 * Animated probability bars for each disease prediction.
 * Color-coded by risk level. Top prediction highlighted with glow.
 *
 * Props:
 *   predictions — array of { disease, probability, status }
 *   topPrediction — { disease, probability, status }
 */
const PredictionResults = ({ predictions, topPrediction }) => {
  if (!predictions || predictions.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="space-y-3"
    >
      <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4 flex items-center gap-2">
        <Trophy className="w-4 h-4 text-primary" />
        Prediction Results
      </h3>

      {predictions.map((pred, i) => {
        const pct = Math.round(pred.probability * 100);
        const isTop = topPrediction && pred.disease === topPrediction.disease && pred.model === topPrediction.model;

        let riskColor, riskBg, riskBorder, riskGlow, Icon, riskLabel;

        if (pred.status === 'high_risk') {
          riskColor = 'text-red-400';
          riskBg = 'bg-red-500';
          riskBorder = 'border-red-500/20';
          riskGlow = 'shadow-[0_0_12px_rgba(239,68,68,0.3)]';
          Icon = ShieldAlert;
          riskLabel = 'High Risk';
        } else if (pred.status === 'medium_risk') {
          riskColor = 'text-amber-400';
          riskBg = 'bg-amber-500';
          riskBorder = 'border-amber-500/20';
          riskGlow = 'shadow-[0_0_12px_rgba(245,158,11,0.2)]';
          Icon = ShieldQuestion;
          riskLabel = 'Medium Risk';
        } else {
          riskColor = 'text-emerald-400';
          riskBg = 'bg-emerald-500';
          riskBorder = 'border-emerald-500/20';
          riskGlow = '';
          Icon = ShieldCheck;
          riskLabel = 'Low Risk';
        }

        return (
          <motion.div
            key={`${pred.model}-${pred.disease}`}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: i * 0.1 + 0.2 }}
            className={`
              p-3.5 rounded-lg border transition-all
              ${isTop ? `${riskBorder} bg-slate-800/80 ${riskGlow}` : 'border-slate-700/30 bg-slate-800/40'}
            `}
          >
            {/* Header row */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Icon className={`w-4 h-4 ${riskColor}`} />
                <span className={`text-sm font-medium ${isTop ? 'text-slate-100' : 'text-slate-300'}`}>
                  {pred.disease}
                </span>
                {isTop && (
                  <span className="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-primary/15 text-primary border border-primary/25">
                    Top
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <span className={`text-xs ${riskColor}`}>{riskLabel}</span>
                <span className={`text-sm font-bold ${riskColor}`}>{pct}%</span>
              </div>
            </div>

            {/* Bar */}
            <div className="w-full h-1.5 bg-slate-700/50 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.8, delay: i * 0.1 + 0.4, ease: 'easeOut' }}
                className={`h-full rounded-full ${riskBg}`}
              />
            </div>
          </motion.div>
        );
      })}
    </motion.div>
  );
};

export default PredictionResults;
