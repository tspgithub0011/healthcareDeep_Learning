// Allowed file types for upload
export const ALLOWED_FILE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

// Maximum file size in bytes (10 MB)
export const MAX_FILE_SIZE = 10 * 1024 * 1024;

// API base URL — use env variable in production, proxy in dev
export const API_URL = import.meta.env.VITE_API_URL || '';

// Risk level thresholds
export const RISK_THRESHOLDS = {
  HIGH: 0.7,
  MEDIUM: 0.4,
};

// Human-readable file type names
export const FILE_TYPE_LABELS = {
  'image/jpeg': 'JPEG',
  'image/png': 'PNG',
  'image/webp': 'WEBP',
};

// Image type display names
export const IMAGE_TYPE_LABELS = {
  xray: 'X-Ray',
  mri: 'MRI',
  skin: 'Skin',
};

// Status labels for risk levels
export const RISK_LABELS = {
  high_risk: { label: 'High Risk', color: 'danger' },
  medium_risk: { label: 'Medium Risk', color: 'warning' },
  low_risk: { label: 'Low Risk', color: 'success' },
};
