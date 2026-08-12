"use client";

import { useEffect, useState } from "react";
import { Samples } from "@/components/screens/Samples";
import { Upload } from "@/components/screens/Upload";
import { Processing } from "@/components/screens/Processing";
import { Review } from "@/components/screens/Review";
import { Result } from "@/components/screens/Result";
import { How } from "@/components/screens/How";
import { THEMES, useTheme } from "@/lib/theme";
import { SAMPLES } from "@/lib/data";
import type { SampleDataset, Screen } from "@/lib/types";

const TABS: { id: Screen; label: string }[] = [
  { id: "samples", label: "Samples" },
  { id: "upload", label: "Upload" },
  { id: "review", label: "Review" },
  { id: "result", label: "Result" },
  { id: "how", label: "How it works" },
];

export default function Page() {
  const { theme, setTheme } = useTheme();
  const [screen, setScreenRaw] = useState<Screen>("samples");
  const [dataset, setDataset] = useState<SampleDataset>(SAMPLES[0]);

  // Mirror screen + dataset into the URL so a view can be linked, reloaded and
  // shared rather than living only in component state.
  const setScreen = (s: Screen, d?: SampleDataset) => {
    setScreenRaw(s);
    const ds = d ?? dataset;
    const q = new URLSearchParams({ screen: s, dataset: ds.id });
    window.history.replaceState(null, "", `?${q}`);
  };

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const s = q.get("screen") as Screen | null;
    const d = SAMPLES.find((x) => x.id === q.get("dataset"));
    if (d) setDataset(d);
    if (s && ["samples","upload","processing","review","result","how"].includes(s)) {
      setScreenRaw(s);
    }
  }, []);
  const [backendUp, setBackendUp] = useState(false);

  // The compute service is a separate local process; the samples path works
  // without it. Probe once so Upload can degrade honestly rather than dangle.
  useEffect(() => {
    const url = process.env.NEXT_PUBLIC_API_URL;
    if (!url) return;
    fetch(`${url}/health`, { signal: AbortSignal.timeout(2500) })
      .then((r) => setBackendUp(r.ok))
      .catch(() => setBackendUp(false));
  }, []);

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 18,
          flexWrap: "wrap",
          padding: "12px clamp(14px, 2.5vw, 28px)",
          borderBottom: "1px solid var(--line)",
          background: "var(--panel)",
        }}
      >
        <div style={{ font: "600 15px/1 var(--mono)", letterSpacing: "-.01em" }}>
          cubit
        </div>

        <nav style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setScreen(t.id)}
              style={{
                border: 0,
                padding: "7px 11px",
                borderRadius: 6,
                font: "500 12.5px/1 var(--sans)",
                background: screen === t.id ? "var(--soft)" : "transparent",
                color: screen === t.id ? "var(--ink)" : "var(--muted)",
              }}
            >
              {t.label}
            </button>
          ))}
        </nav>

        <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
          {THEMES.map((t) => (
            <button
              key={t.id}
              onClick={() => setTheme(t.id)}
              title={t.label}
              aria-label={t.label}
              style={{
                width: 22,
                height: 22,
                borderRadius: 5,
                border:
                  theme === t.id
                    ? "2px solid var(--accent)"
                    : "1px solid var(--line)",
                background: t.swatch,
                cursor: "pointer",
              }}
            />
          ))}
        </div>
      </header>

      <main
        style={{
          flex: 1,
          width: "100%",
          maxWidth: 1180,
          margin: "0 auto",
          padding: "clamp(18px, 3vw, 34px) clamp(14px, 2.5vw, 28px) 60px",
        }}
      >
        {screen === "samples" && (
          <Samples
            onOpen={(d) => {
              setDataset(d);
              setScreen("result", d);
            }}
            onUpload={() => setScreen("upload")}
          />
        )}
        {screen === "upload" && (
          <Upload
            backendUp={backendUp}
            onBack={() => setScreen("samples")}
            onStart={() => setScreen("processing")}
          />
        )}
        {screen === "processing" && (
          <Processing onDone={() => setScreen("review")} />
        )}
        {screen === "review" && (
          <Review
            dataset={dataset}
            onBack={() => setScreen("samples")}
            onConfirm={() => setScreen("result")}
          />
        )}
        {screen === "result" && (
          <Result dataset={dataset} onBack={() => setScreen("samples")} />
        )}
        {screen === "how" && <How />}
      </main>
    </div>
  );
}
