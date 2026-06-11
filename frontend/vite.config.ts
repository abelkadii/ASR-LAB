import { defineConfig } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'


function figmaAssetResolver() {
  return {
    name: 'figma-asset-resolver',
    resolveId(id) {
      if (id.startsWith('figma:asset/')) {
        const filename = id.replace('figma:asset/', '')
        return path.resolve(__dirname, 'src/assets', filename)
      }
    },
  }
}

export default defineConfig({
  plugins: [
    figmaAssetResolver(),
    // The React and Tailwind plugins are both required for Make, even if
    // Tailwind is not being actively used – do not remove them
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      // Alias @ to the src directory
      '@': path.resolve(__dirname, './src'),
    },
  },

  // File types to support raw imports. Never add .css, .tsx, or .ts files to this.
  assetsInclude: ['**/*.svg', '**/*.csv'],
  server: {
    proxy: {
      '/models': 'http://127.0.0.1:5000',
      '/datasets': 'http://127.0.0.1:5000',
      '/custom-datasets': 'http://127.0.0.1:5000',
      '/dataset-samples': 'http://127.0.0.1:5000',
      '/dataset-audio': 'http://127.0.0.1:5000',
      '/progress': 'http://127.0.0.1:5000',
      '/transcribe': 'http://127.0.0.1:5000',
      '/transcribe-sample': 'http://127.0.0.1:5000',
      '/label-audio': 'http://127.0.0.1:5000',
      '/label-sample': 'http://127.0.0.1:5000',
      '/word-variants': 'http://127.0.0.1:5000',
      '/compare': 'http://127.0.0.1:5000',
      '/analyze-batch': 'http://127.0.0.1:5000',
    },
  },
})
