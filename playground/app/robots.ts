import type { MetadataRoute } from "next";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: "*",
      allow: "/",
    },
    sitemap: "https://qala-lang.vercel.app/sitemap.xml",
    host: "https://qala-lang.vercel.app",
  };
}
