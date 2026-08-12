"use client";

import { useEffect, useState } from "react";
import { Label, Panel } from "@/components/ui/primitives";
import { STAGES } from "@/lib/data";

/** Stage-by-stage progress. 45 s is too long for a bare spinner, and the stage
 *  boundaries are natural ticks — see prompt.md §5. */
export function Processing({ onDone }: { onDone: () => void }) {
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    if (current >= STAGES.length) {
      const t = setTimeout(onDone, 400);
      return () => clearTimeout(t);
    }
    // Mock timing proportional to the real measured seconds.
    const t = setTimeout(() => setCurrent((c) => c + 1), STAGES[current].seconds * 90);
    return () => clearTimeout(t);
  }, [current, onDone]);

  return (
    <div className="fadein" style={{ display: "grid", gap: 16, maxWidth: 640 }}>
      <div style={{ font: "500 15px/1 var(--sans)" }}>Processing</div>
      <Panel>
        {STAGES.map((s, i) => {
          const done = i < current;
          const running = i === current;
          return (
            <div
              key={s.n}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "11px 0",
                borderBottom:
                  i < STAGES.length - 1 ? "1px solid var(--line)" : "none",
                opacity: done || running ? 1 : 0.4,
              }}
            >
              <div
                style={{
                  width: 20,
                  height: 20,
                  borderRadius: 5,
                  flexShrink: 0,
                  display: "grid",
                  placeItems: "center",
                  background: done ? "var(--accent)" : "var(--soft)",
                  color: done ? "var(--accent-ink)" : "var(--muted)",
                  font: "500 11px/1 var(--mono)",
                  animation: running ? "blip 1.1s ease-in-out infinite" : undefined,
                }}
              >
                {done ? "✓" : s.n}
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ font: "500 13px/1.3 var(--sans)" }}>{s.label}</div>
                <div
                  style={{
                    font: "400 11.5px/1.4 var(--mono)",
                    color: "var(--muted)",
                    marginTop: 2,
                  }}
                >
                  {s.out}
                </div>
              </div>
              <div
                style={{
                  font: "400 11.5px/1 var(--mono)",
                  color: "var(--muted)",
                }}
              >
                ~{s.seconds}s
              </div>
            </div>
          );
        })}
      </Panel>
      <div
        style={{
          font: "400 11.5px/1.6 var(--sans)",
          color: "var(--muted)",
        }}
      >
        Processing runs on a single local GPU, so uploads are handled one at a
        time. Stage 1 dominates; the rest is geometry.
      </div>
    </div>
  );
}
