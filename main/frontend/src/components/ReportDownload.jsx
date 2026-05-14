import { useState } from 'react';
import { FileDown, Loader2 } from 'lucide-react';
import { downloadReport } from '../services/api';

/**
 * Download PDF report button.
 * Appears after successful prediction.
 *
 * Props:
 *   predictionData — full prediction response from API
 *   previewUrl     — preview URL for original image (for converting to base64)
 */
const ReportDownload = ({ predictionData, previewUrl }) => {
  const [loading, setLoading] = useState(false);

  const handleDownload = async () => {
    if (loading) return;
    setLoading(true);

    try {
      // Convert image preview to base64
      let imageBase64 = '';
      if (previewUrl) {
        const response = await fetch(previewUrl);
        const blob = await response.blob();
        imageBase64 = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result);
          reader.readAsDataURL(blob);
        });
      }

      const pdfBlob = await downloadReport(predictionData, imageBase64);

      // Trigger download
      const url = URL.createObjectURL(pdfBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `healthcare-dl-report-${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      // Silently handle — could add error state here
      console.error('Failed to download report');
    } finally {
      setLoading(false);
    }
  };

  return (
    <button
      onClick={handleDownload}
      disabled={loading}
      className="flex items-center justify-center gap-2 w-full px-4 py-3 rounded-btn
        bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white
        border border-slate-700/50 hover:border-slate-600
        transition-all duration-200 text-sm font-medium
        disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {loading ? (
        <>
          <Loader2 className="w-4 h-4 animate-spin" />
          Generating Report…
        </>
      ) : (
        <>
          <FileDown className="w-4 h-4" />
          Download PDF Report
        </>
      )}
    </button>
  );
};

export default ReportDownload;
