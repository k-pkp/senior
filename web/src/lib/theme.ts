"use client";

import { useEffect, useState } from "react";
import type { ThemeName } from "./types";

export const THEMES: { id: ThemeName; label: string; swatch: string }[] = [
  { id: "clinical", label: "Clinical", swatch: "#f4f2ee" },
  { id: "instrument", label: "Instrument", swatch: "#15181b" },
  { id: "paper", label: "Paper", swatch: "#efe9dd" },
];

// Theme state hook: keeps the chosen theme on the document root element.
export function useTheme() {
  // "instrument" is the dark one. A reconstruction is read against its own
  // surface shading, and a dark ground shows that better than a near-white
  // one — which also washed the limb out against the background.
  const [theme, setTheme] = useState<ThemeName>("instrument");
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);
  return { theme, setTheme };
}
