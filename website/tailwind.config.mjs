/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        apple: {
          blue: '#0A84FF',
          bg: '#0a0a0c',
          panel: 'rgba(255,255,255,0.06)',
        },
        prism: {
          cream: '#FFFBF2',
          purple: '#5E2B97',
          'purple-light': '#7B3FC4',
          lime: '#B8E447',
          'lime-dim': '#8FB832',
        },
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          '"SF Pro Display"',
          '"SF Pro Text"',
          'system-ui',
          'sans-serif',
        ],
        mono: ['"SF Mono"', 'ui-monospace', 'Menlo', 'monospace'],
      },
      boxShadow: {
        glass: '0 8px 32px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.08)',
        'glass-lg': '0 24px 64px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.1)',
        logo: '0 32px 100px rgba(94, 43, 151, 0.45), 0 12px 32px rgba(0,0,0,0.3)',
        glow: '0 0 60px rgba(94, 43, 151, 0.35)',
        'glow-lime': '0 0 40px rgba(184, 228, 71, 0.2)',
      },
      backdropBlur: {
        glass: '40px',
      },
      animation: {
        float: 'float 6s ease-in-out infinite',
        'fade-up': 'fadeUp 0.8s ease-out both',
        shimmer: 'shimmer 3s ease-in-out infinite',
        pulseSlow: 'pulseSlow 4s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px) rotate(-0.5deg)' },
          '50%': { transform: 'translateY(-14px) rotate(0.5deg)' },
        },
        fadeUp: {
          from: { opacity: '0', transform: 'translateY(24px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%, 100%': { opacity: '0.5' },
          '50%': { opacity: '1' },
        },
        pulseSlow: {
          '0%, 100%': { opacity: '0.4', transform: 'scale(1)' },
          '50%': { opacity: '0.7', transform: 'scale(1.05)' },
        },
      },
    },
  },
  plugins: [],
};
