import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { nodePolyfills } from 'vite-plugin-node-polyfills';

export default defineConfig({
  plugins: [
    react(),
    nodePolyfills({
      globals: {
        global: true,
      },
      include: [],
    }),
  ],
  server: {
    host: true,
    port: 5173
  }
});