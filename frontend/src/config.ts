// API Configuration
export const API_CONFIG = {
  BASE_URL: 'http://localhost:8000',
  ENDPOINTS: {
    RECOMMEND: '/recommend',
    HEALTH: '/health',
    ROOT: '/'
  }
};

// Environment settings
export const ENV_CONFIG = {
  DEV_MODE: false, // Set to true for demo data
  API_TIMEOUT: 10000, // 10 seconds
  RETRY_ATTEMPTS: 3
};

// Default form values
export const DEFAULT_FORM_VALUES = {
  movie_title: '',
  user_id: 1,
  num_recommendations: 5
}; 