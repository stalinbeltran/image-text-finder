import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api -> the FastAPI backend so the SPA uses same-origin
// relative URLs. Override the target with ITF_API_URL if needed.
//
// Port 5173 is not a default worth changing: D4 closed CORS to exactly this
// origin (api.md §6). Moving it here means moving it there too.
const API_URL = process.env.ITF_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: API_URL,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
