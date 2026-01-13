/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // ============================================
        // MOMENT COLOR PALETTE
        // Edit these values to change colors globally
        // ============================================

        // Accent color - rust orange for highlights
        accent: {
          DEFAULT: '#C45508',
          hover: '#A34507',
          light: '#E87A2E',
        },

        // Primary action color - buttons, links, interactive elements
        primary: {
          DEFAULT: '#2563eb', // blue-600
          hover: '#1d4ed8',   // blue-700
          light: '#dbeafe',   // blue-100 - for backgrounds
          muted: '#93c5fd',   // blue-300 - for subtle accents
        },

        // UI grays - for buttons, backgrounds, borders
        ui: {
          DEFAULT: '#6b7280', // gray-500
          light: '#f3f4f6',   // gray-100 - button backgrounds
          medium: '#9ca3af',  // gray-400 - icons, secondary text
          dark: '#374151',    // gray-700 - strong text
          border: '#e5e7eb',  // gray-200 - borders
        },
      },
    },
  },
  plugins: [],
}