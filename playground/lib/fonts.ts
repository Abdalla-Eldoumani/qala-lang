// the three codex fonts, self-hosted by next/font at build time.
// each binds a css variable; layout.tsx puts the variables on <html>.
// instrument serif is not a variable font -- weight 400 is required.
// no inter; system-ui appears only as the last-resort fallback in the
// font-family chains defined in globals.css. the choices are locked by
// the ui spec: instrument serif for display, geist for body, jetbrains
// mono for code -- do not substitute.
import { Instrument_Serif, Geist, JetBrains_Mono } from "next/font/google";

export const fontDisplay = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-display",
  display: "swap",
});

export const fontBody = Geist({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-body",
  display: "swap",
});

export const fontMono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
  display: "swap",
});
