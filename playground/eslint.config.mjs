// flat eslint config for next 16. `next lint` was removed in next 16,
// so the lint script calls eslint directly against this file.
// eslint 9.0.0 does not export `eslint/config` (defineConfig and
// globalIgnores were added later in the eslint 9 line), so this is the
// plain flat-array form: spread the two eslint-config-next entry points,
// then a final ignores object.
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = [
  ...nextVitals,
  ...nextTs,
  {
    // the wasm-pack glue under lib/wasm/ is generated; it carries its own
    // eslint-disable banner, but ignoring the directory keeps output clean.
    ignores: [".next/**", "out/**", "build/**", "next-env.d.ts", "lib/wasm/**"],
  },
];

export default eslintConfig;
