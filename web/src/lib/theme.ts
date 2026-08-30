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
  const [theme, setTheme] = useState<ThemeName>("clinical");
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);
  return { theme, setTheme };
}
