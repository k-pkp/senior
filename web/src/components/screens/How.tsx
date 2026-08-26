"use client";

import { Label, Panel } from "@/components/ui/primitives";
import { STAGES } from "@/lib/data";

const DETAIL: Record<number, string> = {
  1: "One forward pass of VGGT-1B produces a 3D point and a confidence value for every pixel of every photo, plus camera poses. This is the only neural step and where all the uncertainty originates.",
  2: "Points are filtered by confidence — about 55% survive — and statistical outliers removed. The result is one coloured cloud of the whole scene: floor, object, cube.",
  3: "The floor plane is removed (without it everything is connected through the ground and cannot be separated), the rest is clustered into objects, and the reference cube is identified by how cube-like and how black-and-white it is. The duplicated surface VGGT emits is collapsed and the survivors projected onto a locally fitted quadratic surface. The marker band is found using the colour Stage 0 measured from your own photographs rather than a fixed threshold, and a plane is fitted through it — which you confirm before anything is cut.",
  4: "The point cloud becomes a triangle mesh by Poisson reconstruction, which follows the points closely. Poisson carries no guarantee that the result is a single solid, so if the repair stage cannot bring the mesh to Euler characteristic 2, this stage runs again with an alpha shape — whose search selects on that property and therefore cannot fail it.",
  5: "PyMeshFix closes the boundary and removes self-intersections and non-manifold edges, then the mesh is checked: closed, and Euler characteristic 2. This is not insurance — it is what makes a Poisson mesh usable, and a mesh that still fails sends Stage 4 back to the alpha shape.",
  6: "A closed mesh has an exact volume: sum the signed tetrahedron volumes over its triangles, no voxel approximation. Real-world size comes from the reference cube — the ratio of its true 2744 cm³ to its measured mesh volume. If a mesh is not closed the stage falls back to flooding a voxel grid, which over-reads and can leak.",
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
        A framing gate reads your photographs first, then six stages run. Only
        the first of them uses a neural network; everything after is geometry.
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
