import axios from "axios";

/**
 * Central Axios instance.
 * - baseURL is empty because Vite proxies /api and /storage to the backend.
 * - a request interceptor attaches the JWT access token.
 * - a response interceptor redirects to /login on 401.
 */
const api = axios.create({ baseURL: "" });

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("access_token");
      localStorage.removeItem("user");
      if (window.location.pathname !== "/login") window.location.href = "/login";
    }
    return Promise.reject(err);
  }
);

// ---------- API calls ----------
export const authApi = {
  register: (data) => api.post("/api/auth/register", data),
  login: (data) => api.post("/api/auth/login", data),
  me: () => api.get("/api/auth/me"),
};

export const detectApi = {
  image: (formData) =>
    api.post("/api/detect/image", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    }),
  video: (formData) =>
    api.post("/api/detect/video", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    }),
  compare: (formData) =>
    api.post("/api/detect/compare", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    }),
};

export const historyApi = {
  list: (params) => api.get("/api/history", { params }),
  get: (id) => api.get(`/api/history/${id}`),
};

export const analyticsApi = {
  summary: () => api.get("/api/analytics/summary"),
};

export const modelsApi = {
  list: () => api.get("/api/models"),
};

export default api;
