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
  /** A mesh that fails to load, as opposed to a CSV that fails to parse. */
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    loadVolumes(dataset.volumesCsv).then(setRows).catch((e) => setErr(String(e)));
  }, [dataset]);

  const scale = useMemo(() => (rows ? linearScale(rows) : null), [rows]);
  // Without a scale nothing can be drawn in centimetres, and drawing it in
  // arbitrary units would label the grid with a lie.
  const scaleError =
    rows && scale === null
      ? "This run was measured by a Stage 6 that does not report the reference "
        + "cube's edge length, so the scene cannot be drawn to scale."
      : null;
  const obj = rows?.find((r) => !r.is_ref);
  const ref = rows?.find((r) => r.is_ref);

  const url = view === "object" ? dataset.meshes.leg : dataset.meshes.scene;
  // Only the reference has been measured: this is a run whose cut has been
  // detected but not yet confirmed, so the object genuinely has no volume yet.
  // Saying that is the honest rendering; the alternative is an empty panel and
  // a download button offering a mesh that does not exist.
  const awaitingCut = rows != null && rows.length > 0 && obj == null;

  if (awaitingCut) {
    return (
      <div className="fadein" style={{ display: "grid", gap: 16, maxWidth: 640 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <Button variant="ghost" onClick={onBack}>← Back</Button>
          <div style={{ font: "500 15px/1 var(--sans)" }}>{dataset.label}</div>
        </div>
        <Panel>
          <Label>Not measured yet</Label>
          <div style={{ font: "400 13px/1.7 var(--sans)", marginTop: 8 }}>
            The scene is reconstructed and the cutting plane is detected, but
            the cut has not been applied — so the object has no volume yet. Open
            <strong> Review</strong>, check where the cut falls, and confirm it.
          </div>
          {ref && (
            <div
              style={{
                font: "400 12px/1.6 var(--mono)",
                color: "var(--muted)",
                marginTop: 10,
              }}
            >
              reference cube measured: {ref.real_vol_cm3.toFixed(0)} cm³ ·{" "}
              {ref.height_cm.toFixed(2)} cm tall
            </div>
          )}
        </Panel>
      </div>
    );
  }

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
            <Viewport error={scaleError ?? loadError}>
              {rows && scale !== null && (
                <MeshView url={url} scale={scale} onLoadError={setLoadError} />
              )}
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
