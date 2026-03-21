// dashboard/vite.config.js
import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  base: '/dashboard/',
  build: {
    outDir: '../dist/dashboard',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
});
