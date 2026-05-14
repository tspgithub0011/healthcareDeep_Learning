import axios from 'axios';
import { API_URL } from '../constants';

const api = axios.create({
  baseURL: API_URL,
  timeout: 60000, // 60s — model inference can be slow
});

/**
 * Upload an image for prediction.
 * @param {File} file - The image file to analyze
 * @returns {Promise<object>} Prediction results
 */
export const predictImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/api/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  return response.data;
};

/**
 * Download a PDF report.
 * @param {object} predictionData - The prediction response from /api/predict
 * @param {string} imageBase64 - Base64-encoded original image
 * @returns {Promise<Blob>} PDF blob
 */
export const downloadReport = async (predictionData, imageBase64) => {
  const response = await api.post(
    '/api/report',
    { prediction: predictionData, image: imageBase64 },
    { responseType: 'blob' }
  );

  return response.data;
};

/**
 * Check backend health.
 * @returns {Promise<object>} Health status
 */
export const checkHealth = async () => {
  const response = await api.get('/api/health');
  return response.data;
};

export default api;
