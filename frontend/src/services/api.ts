import axios, { type AxiosResponse } from 'axios';
import { API_CONFIG, ENV_CONFIG } from '../config';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_CONFIG.BASE_URL,
  timeout: ENV_CONFIG.API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`Making request to: ${config.url}`);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    console.log(`Response received from: ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('Response error:', error);
    return Promise.reject(error);
  }
);

export interface RecommendationRequest {
  movie_title: string;
  user_id: number;
  num_recommendations: number;
}

export interface Recommendation {
  title: string;
  genres: string;
  year?: number;
  match_percentage: number;
  why_recommended: string;
}

export interface RecommendationResponse {
  recommendations: Recommendation[];
  error?: string;
  suggestions?: string[];
}

export class ApiService {
  static async getRecommendations(request: RecommendationRequest): Promise<RecommendationResponse> {
    try {
      const response = await apiClient.post<RecommendationResponse>(
        API_CONFIG.ENDPOINTS.RECOMMEND,
        request
      );
      return response.data;
    } catch (error: any) {
      console.error('API call failed:', error);
      
      if (error.response) {
        // Server responded with error status
        return {
          recommendations: [],
          error: error.response.data?.error || 'Server error occurred',
          suggestions: error.response.data?.suggestions || []
        };
      } else if (error.request) {
        // Network error
        throw new Error(`Could not connect to backend at ${API_CONFIG.BASE_URL}. Please make sure the server is running.`);
      } else {
        // Other error
        throw new Error('An unexpected error occurred');
      }
    }
  }

  static async healthCheck(): Promise<boolean> {
    try {
      const response = await apiClient.get(API_CONFIG.ENDPOINTS.HEALTH);
      return response.data.status === 'healthy';
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }
} 