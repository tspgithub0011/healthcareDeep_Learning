import { motion } from 'framer-motion';
import {
  ScanSearch, Cpu, ShieldCheck, ArrowDown,
  Upload, Brain, BarChart3, FileText,
  Wifi, WifiOff, Loader2,
} from 'lucide-react';
import usePrediction from '../hooks/usePrediction';
import UploadZone from '../components/UploadZone';
import ImagePreview from '../components/ImagePreview';
import LoadingSpinner from '../components/LoadingSpinner';
import PredictionResults from '../components/PredictionResults';
import GradCamView from '../components/GradCamView';
import ErrorBanner from '../components/ErrorBanner';
import ReportDownload from '../components/ReportDownload';
import ModelPerformance from '../components/ModelPerformance';

/**
 * Main single-page layout. Wires all components together via usePrediction state machine.
 */
const Home = () => {
  const {
    state,
    file,
    previewUrl,
    result,
    error,
    backendStatus,
    selectFile,
    submit,
    reset,
    retry,
  } = usePrediction();

  const isIdle = state === 'idle';
  const isFileSelected = state === 'file_selected';
  const isUploading = state === 'uploading';
  const isSuccess = state === 'success';
  const isError = state === 'error';

  return (
    <main className="flex-1 w-full">

      {/* ── Server Status Banner (Render cold start) ── */}
      {backendStatus === 'offline' && (
        <div className="bg-amber-500/10 border-b border-amber-500/15 py-2 px-4">
          <p className="flex items-center justify-center gap-2 text-xs sm:text-sm text-amber-300 font-medium text-center">
            <WifiOff className="w-4 h-4 flex-shrink-0" />
            Server is waking up… First request may take 30–60 seconds.
          </p>
        </div>
      )}
      {backendStatus === 'checking' && (
        <div className="bg-slate-800/50 border-b border-slate-700/30 py-2 px-4">
          <p className="flex items-center justify-center gap-2 text-xs text-slate-400 text-center">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Connecting to server…
          </p>
        </div>
      )}

      {/* ── Hero Section (idle state) ── */}
      {isIdle && (
        <motion.section
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="relative text-center pt-14 sm:pt-20 pb-10 px-4 overflow-hidden"
        >
          {/* Background glows */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[500px] h-[250px] bg-primary/15 rounded-full blur-[100px] pointer-events-none" />
          <div className="absolute top-24 right-0 w-[300px] h-[150px] bg-secondary/10 rounded-full blur-[80px] pointer-events-none" />

          <div className="relative max-w-3xl mx-auto">
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="inline-flex items-center gap-2 px-3.5 py-1.5 mb-6 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs sm:text-sm font-medium"
            >
              <ScanSearch className="w-3.5 h-3.5" />
              Zero-Choice AI Routing
            </motion.div>

            {/* Headline */}
            <motion.h1
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-slate-100 leading-tight tracking-tight mb-4"
            >
              Upload Once.{' '}
              <span className="text-gradient">Get Instant Analysis.</span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-base sm:text-lg text-slate-400 max-w-xl mx-auto mb-10 leading-relaxed"
            >
              Upload any Chest X-Ray, Brain MRI, or Skin Lesion photo. Our system automatically
              detects the modality and runs specialized deep learning models — no choices needed.
            </motion.p>

            {/* Feature pills */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="flex flex-wrap items-center justify-center gap-4 sm:gap-6 mb-10"
            >
              {[
                { icon: ScanSearch, text: '6 Disease Models', color: 'text-primary' },
                { icon: Cpu, text: 'Grad-CAM Heatmaps', color: 'text-secondary' },
                { icon: ShieldCheck, text: 'Privacy First', color: 'text-emerald-400' },
              ].map(({ icon: Icon, text, color }) => (
                <div key={text} className="flex items-center gap-2 text-sm text-slate-400">
                  <Icon className={`w-4 h-4 ${color}`} />
                  {text}
                </div>
              ))}
            </motion.div>
          </div>
        </motion.section>
      )}

      {/* ── How It Works (idle state) ── */}
      {isIdle && (
        <motion.section
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="max-w-4xl mx-auto px-4 sm:px-6 mb-14"
        >
          <h2 className="text-center text-sm font-semibold text-slate-500 uppercase tracking-wider mb-6">
            How It Works
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
            {[
              { icon: Upload, title: 'Upload', desc: 'Drop any medical scan', color: 'text-primary', bg: 'bg-primary/10 border-primary/20' },
              { icon: Brain, title: 'Detect', desc: 'AI identifies modality', color: 'text-secondary', bg: 'bg-secondary/10 border-secondary/20' },
              { icon: BarChart3, title: 'Analyze', desc: '6 disease models run', color: 'text-violet-400', bg: 'bg-violet-500/10 border-violet-500/20' },
              { icon: FileText, title: 'Report', desc: 'Results + Grad-CAM', color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/20' },
            ].map(({ icon: Icon, title, desc, color, bg }, i) => (
              <div
                key={title}
                className={`relative flex flex-col items-center text-center p-4 sm:p-5 rounded-card border ${bg}`}
              >
                <Icon className={`w-6 h-6 ${color} mb-2.5`} />
                <h3 className="text-sm font-semibold text-slate-200 mb-0.5">{title}</h3>
                <p className="text-[11px] sm:text-xs text-slate-400">{desc}</p>
                {/* Step connector */}
                {i < 3 && (
                  <div className="hidden sm:block absolute -right-2.5 top-1/2 -translate-y-1/2 text-slate-700 text-lg">
                    →
                  </div>
                )}
              </div>
            ))}
          </div>
        </motion.section>
      )}

      {/* ── Main Content Area ── */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 pb-16">

        {/* Error banner (shows for any error state) */}
        {isError && error && (
          <div className="mb-6">
            <ErrorBanner message={error} onRetry={retry} onDismiss={reset} />
          </div>
        )}

        {/* Upload Zone (idle state) */}
        {isIdle && (
          <div className="max-w-2xl mx-auto">
            <UploadZone onFileSelect={selectFile} disabled={backendStatus === 'checking'} />
            {/* Server status */}
            {backendStatus === 'ready' && (
              <p className="flex items-center justify-center gap-1.5 mt-4 text-xs text-emerald-500/60">
                <Wifi className="w-3 h-3" />
                Server online
              </p>
            )}
          </div>
        )}

      </section>

      {/* ── Model Performance Metrics Table (idle state) ── */}
      {isIdle && <ModelPerformance />}

      {/* ── Interactive Content Area (non-idle states) ── */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6 pb-16">

        {/* File selected — show preview + Analyze button */}
        {isFileSelected && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="max-w-md mx-auto space-y-5"
          >
            <ImagePreview
              previewUrl={previewUrl}
              fileName={file?.name}
              fileSize={file?.size}
              imageType={null}
              onClear={reset}
            />
            <button
              onClick={submit}
              disabled={backendStatus === 'offline'}
              className="w-full py-3.5 rounded-btn bg-gradient-to-r from-primary to-secondary
                text-white font-semibold text-sm shadow-lg shadow-primary/20
                hover:shadow-primary/30 transition-all duration-200 active:scale-[0.98]
                disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {backendStatus === 'offline' ? 'Server offline — waiting…' : 'Analyze Image'}
            </button>
          </motion.div>
        )}

        {/* Uploading — loading state */}
        {isUploading && (
          <div className="max-w-md mx-auto glass-card p-6">
            <ImagePreview
              previewUrl={previewUrl}
              fileName={file?.name}
              fileSize={file?.size}
              imageType={null}
              showClear={false}
            />
            <LoadingSpinner />
          </div>
        )}

        {/* Success — full results view (aria-live for screen readers) */}
        {isSuccess && result && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            aria-live="polite"
            role="region"
            aria-label="Prediction results"
            className="grid grid-cols-1 lg:grid-cols-2 gap-6"
          >
            {/* ── Left column: Image + Grad-CAM ── */}
            <div className="space-y-4">
              <div className="glass-card p-4">
                <ImagePreview
                  previewUrl={previewUrl}
                  fileName={file?.name}
                  fileSize={file?.size}
                  imageType={result.image_type?.detected}
                  showClear={false}
                />
              </div>

              {result.gradcam?.image && (
                <div className="glass-card p-4">
                  <GradCamView
                    originalUrl={previewUrl}
                    heatmapSrc={result.gradcam.image}
                    modelUsed={result.gradcam.model_used}
                  />
                </div>
              )}
            </div>

            {/* ── Right column: Predictions + Actions ── */}
            <div className="space-y-4">

              {/* Image type classification card with probabilities */}
              {result.image_type && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-card p-4"
                >
                  <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Detected Modality</p>
                  <div className="flex items-center gap-3 mb-3">
                    <span className="text-xl font-bold text-slate-100">
                      {result.image_type.detected?.toUpperCase()}
                    </span>
                    <span className="text-sm text-slate-400">
                      {Math.round(result.image_type.confidence * 100)}% confidence
                    </span>
                    {result.image_type.secondary_type && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400">
                        + {result.image_type.secondary_type.toUpperCase()}
                      </span>
                    )}
                  </div>

                  {/* Low confidence warning */}
                  {result.image_type.low_confidence && (
                    <div className="mb-3 p-2.5 rounded-lg bg-amber-500/8 border border-amber-500/20 flex items-start gap-2">
                      <span className="text-amber-400 text-sm mt-0.5">⚠️</span>
                      <p className="text-xs text-amber-300/80 leading-relaxed">
                        Low confidence modality detection. Results from multiple model routes
                        have been combined for better accuracy.
                      </p>
                    </div>
                  )}

                  {/* Probability breakdown: xray / mri / ct_scan / skin */}
                  {result.image_type.probabilities && (
                    <div className="flex gap-2">
                      {Object.entries(result.image_type.probabilities).map(([type, prob]) => {
                        const pct = Math.round(prob * 100);
                        const isDetected = type === result.image_type.detected;
                        const isSecondary = type === result.image_type.secondary_type;
                        return (
                          <div
                            key={type}
                            className={`flex-1 text-center py-1.5 rounded-lg text-xs border ${
                              isDetected
                                ? 'bg-primary/10 border-primary/25 text-primary font-semibold'
                                : isSecondary
                                  ? 'bg-amber-500/10 border-amber-500/20 text-amber-400 font-medium'
                                  : 'bg-slate-800/50 border-slate-700/30 text-slate-500'
                            }`}
                          >
                            <div className="font-medium">{type.toUpperCase()}</div>
                            <div>{pct}%</div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </motion.div>
              )}

              {/* Disease predictions */}
              <div className="glass-card p-4">
                <PredictionResults
                  predictions={result.predictions}
                  topPrediction={result.top_prediction}
                />
              </div>

              {/* Processing time */}
              {result.processing_time_ms && (
                <p className="text-xs text-slate-500 text-center">
                  Processed in {result.processing_time_ms}ms
                </p>
              )}

              {/* Per-response disclaimer */}
              {result.disclaimer && (
                <div className="p-3 rounded-lg bg-amber-500/5 border border-amber-500/15">
                  <p className="text-xs text-amber-300/70 leading-relaxed text-center">
                    {result.disclaimer}
                  </p>
                </div>
              )}

              {/* Actions */}
              <div className="space-y-2">
                <ReportDownload predictionData={result} previewUrl={previewUrl} />
                <button
                  onClick={reset}
                  className="w-full py-3 rounded-btn bg-slate-800 hover:bg-slate-700 text-primary
                    font-medium text-sm border border-primary/20 transition-colors"
                >
                  New Analysis
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </section>
    </main>
  );
};

export default Home;
