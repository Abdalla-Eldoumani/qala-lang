// copy the wasm artifact into public/ so turbopack serves it at
// /wasm/qala_compiler_bg.wasm. turbopack serves public/ at the site
// root but does not bundle the .wasm the wasm-pack glue resolves
// through import.meta.url, so the file has to be a real public asset.
// the wasm-pack glue stays in lib/wasm/; only the .wasm is copied.
import { copyFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = dirname(fileURLToPath(import.meta.url));
const src = join(root, "..", "lib", "wasm", "qala_compiler_bg.wasm");
const destDir = join(root, "..", "public", "wasm");
const dest = join(destDir, "qala_compiler_bg.wasm");

await mkdir(destDir, { recursive: true });
await copyFile(src, dest);
console.log("copied qala_compiler_bg.wasm to public/wasm/");
