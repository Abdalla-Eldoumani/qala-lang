import type { Metadata, Viewport } from "next";
import { fontDisplay, fontBody, fontMono } from "@/lib/fonts";
import "./globals.css";

const SITE_URL = "https://qala-lang.vercel.app";
const SITE_NAME = "Qala";
const SITE_DESCRIPTION = "a language that means what it says";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: SITE_NAME,
  description: SITE_DESCRIPTION,
  alternates: { canonical: "/" },
  openGraph: {
    type: "website",
    siteName: SITE_NAME,
    url: SITE_URL,
    title: SITE_NAME,
    description: SITE_DESCRIPTION,
  },
  twitter: {
    card: "summary_large_image",
    title: SITE_NAME,
    description: SITE_DESCRIPTION,
  },
};

export const viewport: Viewport = {
  themeColor: "#1a1a1f",
  width: "device-width",
  initialScale: 1,
};

// schema.org structured data so search engines and link unfurlers know
// this page is a free, browser-based developer tool with a versioned
// release and a download URL. type="application/ld+json" tells the
// browser this is data, not script. nothing here ever executes.
const structuredData = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: SITE_NAME,
  description: SITE_DESCRIPTION,
  url: SITE_URL,
  applicationCategory: "DeveloperApplication",
  operatingSystem: "Web Browser",
  softwareVersion: "0.1.0",
  downloadUrl: "https://github.com/Abdalla-Eldoumani/qala-lang/releases/latest",
  codeRepository: "https://github.com/Abdalla-Eldoumani/qala-lang",
  programmingLanguage: ["Rust", "TypeScript"],
  license: "https://opensource.org/licenses/MIT",
  offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${fontDisplay.variable} ${fontBody.variable} ${fontMono.variable}`}
    >
      <body>
        <script id="ld-json" type="application/ld+json">
          {JSON.stringify(structuredData)}
        </script>
        {children}
      </body>
    </html>
  );
}
