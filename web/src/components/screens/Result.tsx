"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import { Button, Caveat, Label, Panel, Stat } from "@/components/ui/primitives";
import { linearScale, loadVolumes, REFERENCE_CM } from "@/lib/data";
import type { SampleDataset, VolumeRow } from "@/lib/types";

const Viewport = dynamic(
  () => import("@/components/three/Viewport").then((m) => m.Viewport),
  { ssr: false },
);
const MeshView = dynamic(
  () => import("@/components/three/MeshView").then((m) => m.MeshView),
  { ssr: false },
);

type View = "object" | "scene";

export function Result({
  dataset,
  onBack,
}: {
  dataset: SampleDataset;
  onBack: () => void;
}) {
  const [rows, setRows] = useState<VolumeRow[] | null>(null);
  const [view, setView] = useState<View>("object");
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    loadVolumes(dataset.volumesCsv).then(setRows).catch((e) => setErr(String(e)));
  }, [dataset]);

  const scale = useMemo(() => (rows ? linearScale(rows) : 1), [rows]);
  const obj = rows?.find((r) => !r.is_ref);
  const ref = rows?.find((r) => r.is_ref);

  const url = view === "object" ? dataset.meshes.leg : dataset.meshes.scene;

  return (
    <div className="fadein" style={{ display: "grid", gap: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <Button variant="ghost" onClick={onBack}>
          ← Back
        </Button>
        <div style={{ font: "500 15px/1 var(--sans)" }}>
          {dataset.label}
          <span style={{ color: "var(--muted)", fontWeight: 400 }}>
            {" "}
            · {dataset.subject}
          </span>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0,1fr) 320px",
          gap: 16,
          alignItems: "start",
        }}
        className="result-grid"
      >
        {/* 3D — the centrepiece */}
        <Panel pad={0} style={{ overflow: "hidden" }}>
          <div style={{ height: "clamp(380px, 58vh, 620px)" }}>
            <Viewport>
              {rows && <MeshView url={url} scale={scale} />}
            </Viewport>
          </div>
          <div
            style={{
              display: "flex",
              gap: 8,
              padding: 12,
              borderTop: "1px solid var(--line)",
              alignItems: "center",
              flexWrap: "wrap",
            }}
          >
            {(["object", "scene"] as View[]).map((v) => (
              <Button
                key={v}
                variant={view === v ? "primary" : "ghost"}
                onClick={() => setView(v)}
                style={{ padding: "7px 12px", fontSize: 12 }}
              >
                {v === "object" ? "Object only" : "With reference cube"}
              </Button>
            ))}
            <div
              style={{
                marginLeft: "auto",
                font: "400 11.5px/1.4 var(--sans)",
                color: "var(--muted)",
              }}
            >
              1 grid square = 1 cm · drag to orbit
            </div>
          </div>
        </Panel>

        {/* Numbers */}
        <div style={{ display: "grid", gap: 12 }}>
          <Panel>
            {err && (
              <div style={{ color: "var(--warn)", font: "400 12px/1.5 var(--sans)" }}>
                {err}
              </div>
            )}
            {obj && (
              <>
                <Stat
                  label="Measured volume"
                  value={obj.real_vol_cm3.toFixed(1)}
                  unit="cm³"
                  big
                />
                <div style={{ marginTop: 4, marginBottom: 16 }}>
                  <span
                    style={{
                      font: "400 13px/1 var(--mono)",
                      color: "var(--muted)",
                    }}
                  >
                    = {obj.real_vol_L.toFixed(3)} L
                  </span>
                </div>

                <Label>Height × width × depth (oriented)</Label>
                <div
                  style={{
                    font: "500 15px/1.5 var(--mono)",
                    marginTop: 6,
                  }}
                >
                  {obj.height_cm.toFixed(2)} × {obj.width_cm.toFixed(2)} ×{" "}
                  {obj.depth_cm.toFixed(2)}
                  <span
                    style={{
                      font: "400 12px/1 var(--mono)",
                      color: "var(--muted)",
                    }}
                  >
                    {" "}
                    cm
                  </span>
                </div>
                <div
                  style={{
                    font: "400 11.5px/1.5 var(--sans)",
                    color: "var(--muted)",
                    marginTop: 6,
                  }}
                >
                  Oriented bounding box, not axis-aligned — an AABB around a
                  tilted object reports its diagonal.
                </div>
              </>
            )}
          </Panel>

          {/* The live self-check. See prompt.md §7. */}
          {ref && (
            <Panel>
              <Label>Reference check</Label>
              <div
                style={{
                  display: "flex",
                  alignItems: "baseline",
                  gap: 8,
                  marginTop: 8,
                }}
              >
                <span style={{ font: "500 20px/1 var(--mono)" }}>
                  {ref.real_vol_cm3.toFixed(0)}
                </span>
                <span
                  style={{ font: "400 13px/1 var(--mono)", color: "var(--muted)" }}
                >
                  vs {(REFERENCE_CM ** 3).toFixed(0)} cm³ nominal
                </span>
              </div>
              <div
                style={{
                  font: "400 11.5px/1.55 var(--sans)",
                  color: "var(--muted)",
                  marginTop: 8,
                }}
              >
                The {REFERENCE_CM} cm cube reconstructed and measured the same
                way as the object. Scale comes from its measured edge length, so
                its volume is free to disagree — this gap is the system's own
                error, not something forced to zero.
              </div>
            </Panel>
          )}

          <Panel>
            <Label>Download</Label>
            <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
              <a href={dataset.meshes.leg} download>
                <Button variant="ghost">.ply</Button>
              </a>
              <a href={dataset.meshes.leg.replace(".ply", ".stl")} download>
                <Button variant="ghost">.stl</Button>
              </a>
            </div>
          </Panel>

          {dataset.nominalMl && (
            <Caveat>
              This object is labelled {dataset.nominalMl} ml, but that is its{" "}
              <em>fill</em> volume — the pipeline measures external displacement,
              which is a larger quantity. The two are not directly comparable, so
              no error percentage is quoted.
            </Caveat>
          )}
        </div>
      </div>

      <style>{`
        @media (max-width: 900px) {
          .result-grid { grid-template-columns: minmax(0,1fr) !important; }
        }
      `}</style>
    </div>
  );
}
