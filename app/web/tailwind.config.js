/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    // Override Tailwind's default gray with warm neutrals globally
    colors: {
      // Preserve all Tailwind defaults we need
      transparent: 'transparent',
      current: 'currentColor',
      black: '#1A1A18',
      white: '#FFFFFF',

      // Warm gray — replaces Tailwind's cool blue-gray everywhere
      gray: {
        50:  '#FAFAF8',
        100: '#F3F3EF',
        200: '#E8E8E3',
        300: '#D4D4CE',
        400: '#8A8A82',
        500: '#8A8A82',
        600: '#5C5C56',
        700: '#5C5C56',
        800: '#3A3A36',
        900: '#1A1A18',
        950: '#0F0F0D',
      },

      // Semantic colors Tailwind expects
      red:    { 50: '#fef2f2', 100: '#fee2e2', 200: '#fecaca', 300: '#fca5a5', 400: '#f87171', 500: '#ef4444', 600: '#dc2626', 700: '#b91c1c', 800: '#991b1b', 900: '#7f1d1d' },
      orange: { 50: '#fff7ed', 100: '#ffedd5', 200: '#fed7aa', 300: '#fdba74', 400: '#fb923c', 500: '#f97316', 600: '#ea580c', 700: '#c2410c', 800: '#9a3412', 900: '#7c2d12' },
      yellow: { 50: '#fefce8', 100: '#fef9c3', 200: '#fef08a', 300: '#fde047', 400: '#facc15', 500: '#eab308', 600: '#ca8a04', 700: '#a16207', 800: '#854d0e', 900: '#713f12' },
      green:  { 50: '#f0fdf4', 100: '#dcfce7', 200: '#bbf7d0', 300: '#86efac', 400: '#4ade80', 500: '#22c55e', 600: '#16a34a', 700: '#15803d', 800: '#166534', 900: '#14532d' },
      blue:   { 50: '#eff6ff', 100: '#dbeafe', 200: '#bfdbfe', 300: '#93c5fd', 400: '#60a5fa', 500: '#3b82f6', 600: '#2563eb', 700: '#1d4ed8', 800: '#1e40af', 900: '#1e3a8a' },
      purple: { 50: '#faf5ff', 100: '#f3e8ff', 200: '#e9d5ff', 300: '#d8b4fe', 400: '#c084fc', 500: '#a855f7', 600: '#9333ea', 700: '#7e22ce', 800: '#6b21a8', 900: '#581c87' },
      stone:  { 50: '#fafaf9', 100: '#f5f5f4', 200: '#e7e5e4', 300: '#d6d3d1', 400: '#a8a29e', 500: '#78716c', 600: '#57534e', 700: '#44403c', 800: '#292524', 900: '#1c1917' },
    },
    extend: {
      keyframes: {
        slideUp: {
          '0%': { transform: 'translateY(100%)' },
          '100%': { transform: 'translateY(0)' },
        },
        blink: {
          '0%':   { transform: 'scaleY(1)', backgroundColor: '#FFFFFF' },
          '40%':  { transform: 'scaleY(0.12)', backgroundColor: '#D97706' },
          '100%': { transform: 'scaleY(1)', backgroundColor: '#FFFFFF' },
        },
      },
      animation: {
        'slide-up': 'slideUp 0.3s ease-out',
        'blink': 'blink 0.3s ease-in-out',
      },
      colors: {
        // ============================================
        // MOMENT COLOR PALETTE — Amber + Warm Neutrals
        // ============================================

        // Amber accent — highlights, progress bars, active states
        accent: {
          DEFAULT: '#D97706',
          hover: '#B45309',
          light: '#FEF3C7',
        },

        // Primary action color — buttons, interactive elements
        primary: {
          DEFAULT: '#5C5C56',
          hover: '#3A3A36',
          light: '#F3F3EF',
          muted: '#8A8A82',
        },

        // UI grays — backward-compatible keys, warm neutral values
        ui: {
          DEFAULT: '#8A8A82',
          light: '#F3F3EF',
          medium: '#8A8A82',
          dark: '#5C5C56',
          border: '#E8E8E3',
        },

        // Warm neutrals — extended palette
        neutral: {
          50:  '#FAFAF8',  // page background
          100: '#F3F3EF',  // card/button backgrounds
          200: '#E8E8E3',  // borders, dividers
          400: '#8A8A82',  // secondary text, icons
          600: '#5C5C56',  // body text
          900: '#1A1A18',  // headings, primary text
        },
      },
    },
  },
  plugins: [],
}