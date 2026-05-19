import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async headers() {
    return [
      {
        // the compiled qala wasm module. it is content-stable per build
        // (a new build is a new file), so it caches aggressively. the
        // explicit Content-Type guarantees WebAssembly.instantiateStreaming
        // gets application/wasm regardless of the server's mime defaults.
        source: "/wasm/qala_compiler_bg.wasm",
        headers: [
          { key: "Content-Type", value: "application/wasm" },
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
