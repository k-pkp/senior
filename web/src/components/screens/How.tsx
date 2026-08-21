"use client";

import { Label, Panel } from "@/components/ui/primitives";
import { STAGES } from "@/lib/data";

const DETAIL: Record<number, string> = {
  1: "One forward pass of VGGT-1B produces a 3D point and a confidence value for every pixel of every photo, plus camera poses. This is the only neural step and where all the uncertainty originates.",
  2: "Points are filtered by confidence — about 55% survive — and statistical outliers removed. The result is one coloured cloud of the whole scene: floor, object, cube.",
  3: "The floor plane is removed (without it everything is connected through the ground and cannot be separated), the rest is clustered into objects, and the reference cube is identified by how cube-like and how black-and-white it is. The coloured marker band is found by colour thresholding and a plane fitted through it.",
  4: "The point cloud becomes a triangle mesh using alpha shapes, choosing the tightest alpha that still produces a closed surface. Smooth surface fitting was tried and rounds sharp rims inward, losing real volume.",
  5: "The mesh is checked for closure and repaired if needed. With alpha shapes it is always already closed, so this is insurance rather than a processing step.",
  6: "Volume is computed by integrating the closed surface exactly — no voxel approximation — then converted to centimetres using the reference cube's measured edge length. A voxel occupancy count runs alongside as an independent cross-check.",
};

export function How() {
  return (
    <div className="fadein" style={{ display: "grid", gap: 16, maxWidth: 760 }}>
      <h2 style={{ font: "500 22px/1.2 var(--sans)", margin: 0 }}>
        How it works
      </h2>
      <p
        style={{
          font: "400 14px/1.65 var(--sans)",
          color: "var(--muted)",
          margin: 0,
        }}
      >
        Six stages. Only the first runs a neural network; everything after is
        geometry.
      </p>

      {STAGES.map((s) => (
        <Panel key={s.n}>
          <div style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
            <span
              style={{
                font: "500 12px/1 var(--mono)",
                color: "var(--accent)",
              }}
            >
              {s.n}
            </span>
            <div style={{ flex: 1 }}>
              <div style={{ font: "500 14px/1.3 var(--sans)" }}>{s.label}</div>
              <p
                style={{
                  font: "400 13px/1.65 var(--sans)",
                  color: "var(--muted)",
                  margin: "7px 0 0",
                }}
              >
                {DETAIL[s.n]}
              </p>
            </div>
          </div>
        </Panel>
      ))}

      <Panel>
        <Label>What this cannot yet do</Label>
        <p
          style={{
            font: "400 13px/1.7 var(--sans)",
            color: "var(--muted)",
            margin: "10px 0 0",
          }}
        >
          Scale cannot be validated. With only one object of known size, the cube
          defines the scale — so it can never disagree with itself. Confirming
          accuracy needs a second known object: calibrate on the cube, predict the
          second object&apos;s size, and compare against a caliper measurement.
          That measurement does not exist yet.
        </p>
      </Panel>
    </div>
  );
}
