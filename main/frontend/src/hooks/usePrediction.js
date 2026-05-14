import { useState, useCallback, useEffect } from 'react';
import { predictImage, checkHealth } from '../services/api';
import { ALLOWED_FILE_TYPES, MAX_FILE_SIZE } from '../constants';

/**
 * State machine hook for the upload → loading → result flow.
 *
 * States: idle → file_selected → uploading → success | error
 */
const usePrediction = () => {
  const [state, setState] = useState('idle');
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [backendStatus, setBackendStatus] = useState('checking'); // checking | ready | offline

  /** Check backend health on mount */
  useEffect(() => {
    let cancelled = false;

    const ping = async () => {
      try {
        const data = await checkHealth();
        if (!cancelled) {
          setBackendStatus(data.status === 'ready' ? 'ready' : 'offline');
        }
      } catch {
        if (!cancelled) setBackendStatus('offline');
      }
    };

    ping();

    // Re-check every 30s if offline (Render cold start recovery)
    const interval = setInterval(() => {
      if (backendStatus === 'offline') ping();
    }, 30000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [backendStatus]);

  /** Validate and select a file */
  const selectFile = useCallback((selectedFile) => {
    setResult(null);

    if (!ALLOWED_FILE_TYPES.includes(selectedFile.type)) {
      setError('Invalid file format. Please upload a JPEG, PNG, or WEBP image.');
      setState('error');
      return;
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      setError(`File too large. Maximum size is ${MAX_FILE_SIZE / (1024 * 1024)} MB.`);
      setState('error');
      return;
    }

    setError('');
    // Revoke previous URL to avoid memory leaks
    if (previewUrl) URL.revokeObjectURL(previewUrl);

    const url = URL.createObjectURL(selectedFile);
    setFile(selectedFile);
    setPreviewUrl(url);
    setState('file_selected');
  }, [previewUrl]);

  /** Submit the file for prediction */
  const submit = useCallback(async () => {
    if (!file) return;

    setState('uploading');
    setError('');

    try {
      const data = await predictImage(file);
      setResult(data);
      setState('success');
    } catch (err) {
      const message =
        err.response?.data?.detail ||
        err.message ||
        'An unexpected error occurred. Please try again.';
      setError(message);
      setState('error');
    }
  }, [file]);

  /** Reset everything back to idle */
  const reset = useCallback(() => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl('');
    setResult(null);
    setError('');
    setState('idle');
  }, [previewUrl]);

  /** Go back from error to file_selected or idle */
  const retry = useCallback(() => {
    setError('');
    if (file) {
      setState('file_selected');
    } else {
      setState('idle');
    }
  }, [file]);

  return {
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
  };
};

export default usePrediction;
