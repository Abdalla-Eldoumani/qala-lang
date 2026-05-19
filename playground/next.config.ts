import type { NextConfig } from "next";

// security headers applied to every response. these are the conventional
// defense-in-depth set: enforce HTTPS, refuse MIME sniffing, refuse to be
// framed, leak only the origin on cross-origin navigation, and lock down
// browser features the playground does not use.
//
// the CSP is intentionally permissive about inline scripts and styles --
// Next.js's hydration scripts and Monaco's runtime styling effectively
// require this without a per-request nonce middleware. the playground
// renders no user-generated content, so the XSS surface is minimal in
// practice. 'wasm-unsafe-eval' is what lets WebAssembly.instantiate run.
const securityHeaders = [
  {
    key: "Strict-Transport-Security",
    value: "max-age=63072000; includeSubDomains; preload",
  },
  {
    key: "X-Content-Type-Options",
    value: "nosniff",
  },
  {
    key: "X-Frame-Options",
    value: "DENY",
  },
  {
    key: "Referrer-Policy",
    value: "strict-origin-when-cross-origin",
  },
  {
    key: "Permissions-Policy",
    value: "camera=(), microphone=(), geolocation=(), interest-cohort=()",
  },
  {
    key: "Content-Security-Policy",
    value: [
      "default-src 'self'",
      // @monaco-editor/react loads the editor runtime, workers, and AMD
      // language packs from cdn.jsdelivr.net. without this allowance the
      // editor sits on its "Loading..." placeholder forever.
      "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' https://cdn.jsdelivr.net",
      "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
      "font-src 'self' data:",
      "img-src 'self' data:",
      "connect-src 'self' https://cdn.jsdelivr.net",
      "worker-src 'self' blob: https://cdn.jsdelivr.net",
      "frame-ancestors 'none'",
      "base-uri 'self'",
      "form-action 'self'",
      "object-src 'none'",
      "upgrade-insecure-requests",
    ].join("; "),
  },
];

const nextConfig: NextConfig = {
  async headers() {
    return [
      {
        source: "/:path*",
        headers: securityHeaders,
      },
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
