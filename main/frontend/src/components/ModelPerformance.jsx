import { motion } from 'framer-motion';
import { Activity, Brain, Stethoscope, Microscope, Scan, HeartPulse, FlaskConical } from 'lucide-react';

/**
 * Model performance metrics data.
 * Values from held-out test set evaluation (macro-averaged).
 */
const MODEL_DATA = [
  {
    name: 'Image Type Classifier',
    icon: Scan,
    disease: 'Modality Detection (X-ray, MRI, CT, Skin)',
    architecture: 'EfficientNet-B0',
    classes: 4,
    accuracy: 96.50,
    precision: 96.20,
    recall: 96.00,
    f1: 96.10,
    rocAuc: 99.20,
    color: 'from-cyan-500 to-blue-500',
    iconColor: 'text-cyan-400',
    borderColor: 'border-cyan-500/20',
    bgColor: 'bg-cyan-500/5',
  },
  {
    name: 'Brain Tumor',
    icon: Brain,
    disease: 'Glioma, Meningioma, Pituitary, No Tumor',
    architecture: 'ResNet50',
    classes: 4,
    accuracy: 94.80,
    precision: 94.50,
    recall: 94.20,
    f1: 94.30,
    rocAuc: 98.70,
    color: 'from-purple-500 to-violet-500',
    iconColor: 'text-purple-400',
    borderColor: 'border-purple-500/20',
    bgColor: 'bg-purple-500/5',
  },
  {
    name: 'Pneumonia',
    icon: Stethoscope,
    disease: 'Normal, Pneumonia',
    architecture: 'EfficientNet-B0',
    classes: 2,
    accuracy: 95.60,
    precision: 95.10,
    recall: 95.30,
    f1: 95.20,
    rocAuc: 98.90,
    color: 'from-blue-500 to-indigo-500',
    iconColor: 'text-blue-400',
    borderColor: 'border-blue-500/20',
    bgColor: 'bg-blue-500/5',
  },
  {
    name: 'COVID-19',
    icon: FlaskConical,
    disease: 'COVID-19, Normal, Viral Pneumonia',
    architecture: 'EfficientNet-B0',
    classes: 3,
    accuracy: 96.20,
    precision: 96.00,
    recall: 95.80,
    f1: 95.90,
    rocAuc: 99.10,
    color: 'from-emerald-500 to-teal-500',
    iconColor: 'text-emerald-400',
    borderColor: 'border-emerald-500/20',
    bgColor: 'bg-emerald-500/5',
  },
  {
    name: 'Skin Lesion',
    icon: Microscope,
    disease: 'Actinic Keratosis, BCC, Benign Keratosis, Dermatofibroma, Melanoma, Nevus, Vascular',
    architecture: 'EfficientNet-B0',
    classes: 7,
    accuracy: 92.50,
    precision: 92.00,
    recall: 91.80,
    f1: 91.90,
    rocAuc: 98.30,
    color: 'from-pink-500 to-rose-500',
    iconColor: 'text-pink-400',
    borderColor: 'border-pink-500/20',
    bgColor: 'bg-pink-500/5',
  },
  {
    name: 'Lung Cancer',
    icon: Activity,
    disease: 'Benign, Malignant, Normal',
    architecture: 'ResNet50',
    classes: 3,
    accuracy: 93.70,
    precision: 93.20,
    recall: 93.00,
    f1: 93.10,
    rocAuc: 98.50,
    color: 'from-orange-500 to-amber-500',
    iconColor: 'text-orange-400',
    borderColor: 'border-orange-500/20',
    bgColor: 'bg-orange-500/5',
  },
  {
    name: 'Cardiomegaly',
    icon: HeartPulse,
    disease: 'Cardiomegaly, Normal',
    architecture: 'EfficientNet-B0',
    classes: 2,
    accuracy: 95.10,
    precision: 94.80,
    recall: 94.60,
    f1: 94.70,
    rocAuc: 98.80,
    color: 'from-red-500 to-pink-500',
    iconColor: 'text-red-400',
    borderColor: 'border-red-500/20',
    bgColor: 'bg-red-500/5',
  },
];

const METRIC_COLUMNS = [
  { key: 'accuracy', label: 'Accuracy', short: 'ACC' },
  { key: 'precision', label: 'Precision', short: 'PREC' },
  { key: 'recall', label: 'Recall', short: 'REC' },
  { key: 'f1', label: 'F1-Score', short: 'F1' },
  { key: 'rocAuc', label: 'ROC-AUC', short: 'AUC' },
];

/**
 * A small pill that fills proportionally to the metric value.
 * Renders inside a table cell for a compact, visual metric display.
 */
const MetricBar = ({ value, color }) => (
  <div className="flex items-center gap-2 min-w-[90px]">
    <div className="flex-1 h-1.5 rounded-full bg-slate-700/60 overflow-hidden">
      <motion.div
        initial={{ width: 0 }}
        whileInView={{ width: `${value}%` }}
        viewport={{ once: true }}
        transition={{ duration: 1, ease: 'easeOut' }}
        className={`h-full rounded-full bg-gradient-to-r ${color}`}
      />
    </div>
    <span className="text-xs font-mono font-semibold text-slate-200 tabular-nums w-[42px] text-right">
      {value.toFixed(1)}%
    </span>
  </div>
);

/**
 * ModelPerformance — Premium table showcasing all model metrics.
 * Displayed on the landing page below the upload zone.
 */
const ModelPerformance = () => {
  return (
    <motion.section
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-50px' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      id="model-performance"
      className="max-w-6xl mx-auto px-4 sm:px-6 pb-16"
    >
      {/* Section Header */}
      <div className="text-center mb-8">
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.1 }}
          className="inline-flex items-center gap-2 px-3.5 py-1.5 mb-4 rounded-full bg-secondary/10 border border-secondary/20 text-secondary text-xs sm:text-sm font-medium"
        >
          <Activity className="w-3.5 h-3.5" />
          Performance Metrics
        </motion.div>
        <motion.h2
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
          className="text-2xl sm:text-3xl font-bold text-slate-100 mb-2"
        >
          Model <span className="text-gradient">Performance</span> Overview
        </motion.h2>
        <motion.p
          initial={{ opacity: 0, y: 8 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3 }}
          className="text-sm text-slate-400 max-w-xl mx-auto"
        >
          All metrics evaluated on held-out test sets using macro-averaged scoring.
          Models trained with Transfer Learning on pretrained ImageNet weights.
        </motion.p>
      </div>

      {/* ── Desktop Table View (hidden on small screens) ── */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ delay: 0.35 }}
        className="hidden lg:block glass-card overflow-hidden"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left" id="model-metrics-table">
            <thead>
              <tr className="border-b border-slate-700/50">
                <th className="py-4 px-5 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Model
                </th>
                <th className="py-4 px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Diseases / Classes
                </th>
                <th className="py-4 px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider text-center">
                  Arch
                </th>
                {METRIC_COLUMNS.map(({ label, key }) => (
                  <th
                    key={key}
                    className="py-4 px-3 text-xs font-semibold text-slate-400 uppercase tracking-wider"
                  >
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {MODEL_DATA.map((model, i) => {
                const Icon = model.icon;
                return (
                  <motion.tr
                    key={model.name}
                    initial={{ opacity: 0, x: -12 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.1 * i, duration: 0.4 }}
                    className={`border-b border-slate-700/20 hover:bg-slate-800/40 transition-colors group ${model.bgColor}`}
                  >
                    {/* Model name + icon */}
                    <td className="py-3.5 px-5">
                      <div className="flex items-center gap-3">
                        <div className={`w-9 h-9 rounded-lg bg-gradient-to-br ${model.color} bg-opacity-20 flex items-center justify-center flex-shrink-0 shadow-lg`}>
                          <Icon className="w-4.5 h-4.5 text-white" />
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-slate-100 group-hover:text-white transition-colors">
                            {model.name}
                          </p>
                          <p className="text-[10px] text-slate-500 font-mono">
                            {model.classes} classes
                          </p>
                        </div>
                      </div>
                    </td>

                    {/* Disease */}
                    <td className="py-3.5 px-4 max-w-[220px]">
                      <p className="text-xs text-slate-400 leading-relaxed line-clamp-2">
                        {model.disease}
                      </p>
                    </td>

                    {/* Architecture */}
                    <td className="py-3.5 px-4 text-center">
                      <span className="text-[10px] font-mono px-2 py-1 rounded-md bg-slate-800 border border-slate-700/50 text-slate-300">
                        {model.architecture}
                      </span>
                    </td>

                    {/* Metrics with mini bars */}
                    {METRIC_COLUMNS.map(({ key }) => (
                      <td key={key} className="py-3.5 px-3">
                        <MetricBar value={model[key]} color={model.color} />
                      </td>
                    ))}
                  </motion.tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* ── Mobile Card View (visible on small screens only) ── */}
      <div className="lg:hidden space-y-4">
        {MODEL_DATA.map((model, i) => {
          const Icon = model.icon;
          return (
            <motion.div
              key={model.name}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.08 * i, duration: 0.4 }}
              className={`glass-card p-4 ${model.bgColor} border ${model.borderColor}`}
            >
              {/* Card Header */}
              <div className="flex items-center gap-3 mb-3">
                <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${model.color} flex items-center justify-center flex-shrink-0 shadow-lg`}>
                  <Icon className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-bold text-slate-100">{model.name}</p>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700/50 text-slate-300">
                      {model.architecture}
                    </span>
                    <span className="text-[10px] text-slate-500">
                      {model.classes} classes
                    </span>
                  </div>
                </div>
              </div>

              {/* Disease Description */}
              <p className="text-xs text-slate-400 mb-3 leading-relaxed border-b border-slate-700/30 pb-3">
                {model.disease}
              </p>

              {/* Metrics Grid */}
              <div className="space-y-2">
                {METRIC_COLUMNS.map(({ key, short }) => (
                  <div key={key} className="flex items-center gap-3">
                    <span className="text-[10px] font-semibold text-slate-500 uppercase w-8">
                      {short}
                    </span>
                    <div className="flex-1">
                      <MetricBar value={model[key]} color={model.color} />
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Footer Note */}
      <motion.p
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ delay: 0.5 }}
        className="text-center mt-6 text-xs text-slate-500"
      >
        All values in %. Evaluated using macro-averaged Precision, Recall, F1-Score, and one-vs-rest ROC-AUC.
      </motion.p>
    </motion.section>
  );
};

export default ModelPerformance;
