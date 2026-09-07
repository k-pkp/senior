"use client";

import { Button, Label, Panel } from "@/components/ui/primitives";
import { SAMPLES } from "@/lib/data";
import type { SampleDataset } from "@/lib/types";

// Landing screen: pick a bundled sample capture or start an upload.
export function Samples({
  onOpen,
  onUpload,
}: {
  onOpen: (d: SampleDataset) => void;
  onUpload: () => void;
}) {
  return (
    <div className="fadein" style={{ display: "grid", gap: 18 }}>
      <div>
        <h1 style={{ font: "500 26px/1.2 var(--sans)", margin: "0 0 8px" }}>
          Volume from photographs
        </h1>
        <p
          style={{
            font: "400 14px/1.6 var(--sans)",
            color: "var(--muted)",
            margin: 0,
            maxWidth: 620,
          }}
        >
          Photograph an object beside a cube of known size. A vision transformer
          reconstructs the scene in 3D from those images alone — no depth sensor,
          no turntable — and the cube converts the result into centimetres.
        </p>
      </div>

      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <Button variant="primary" onClick={onUpload}>
          Upload a photo set
        </Button>
      </div>

      <div>
        <Label>Precomputed runs</Label>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
            gap: 12,
            marginTop: 10,
          }}
        >
          {SAMPLES.map((d) => (
            <Panel key={d.id} pad={14}>
              <div style={{ font: "500 15px/1.3 var(--sans)" }}>{d.label}</div>
              <div
                style={{
                  font: "400 12.5px/1.5 var(--sans)",
                  color: "var(--muted)",
                  marginTop: 4,
                }}
              >
                {d.subject} · {d.frames} photos
              </div>
              <Button
                variant="ghost"
                onClick={() => onOpen(d)}
                style={{ marginTop: 12, width: "100%" }}
              >
                Open result
              </Button>
            </Panel>
          ))}
        </div>
        <div
          style={{
            font: "400 11.5px/1.6 var(--sans)",
            color: "var(--muted)",
            marginTop: 10,
          }}
        >
          These are real pipeline outputs and need no server. Uploading new
          photos requires the compute machine to be reachable.
        </div>
      </div>
    </div>
  );
}
