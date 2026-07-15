import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api -> the FastAPI backend so the SPA can use
// same-origin relative URLs. Override the target with ITF_API_URL if needed.
const API_URL = process.env.ITF_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API_URL,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
