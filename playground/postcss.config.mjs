// next still needs a thin postcss file for tailwind 4; the zero-config
// story is vite-only. turbopack picks this up automatically.
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
};

export default config;
