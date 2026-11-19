import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    watch: { ignored: ['**/data/**', '**/services/**', '**/ml/**', '**/tests/**', '**/.opencode/**', '**/.impeccable/**', '**/.venv/**'] },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8100',
        ws: true,
        changeOrigin: false,
      },
    },
  },
});
