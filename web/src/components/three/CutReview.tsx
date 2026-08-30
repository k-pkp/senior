"use client";

import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { Bounds } from "@react-three/drei";
import type { CutPlane } from "@/lib/types";
import { fitSlice, type CrossSectionResult } from "@/lib/crosssection";
import { dirToScene, pointToScene, usePly } from "./usePly";

/** At most two cuts bound a segment; see segmentation.py:MAX_MARKERS. */
export const MAX_PLANES = 2;

/**
 * Apply the pipeline's cut rule client-side so the split updates while dragging.
 *
 * Mirrors core/segmentation.py:apply_marker_cut exactly:
 *   0 planes -> no cut
 *   1 plane  -> keep what is BELOW it
 *   2 planes -> keep what is BETWEEN them
 *
 * Below/between is measured along each plane's own normal, flipped to point up
 * first so the detected normal's sign cannot change the outcome. Scene space is
 * Y-up, so "up" here is +Y.
 *
 * A dot product per point per plane; at a few thousand points this is far below
 * frame budget, so no need to debounce.
 */
// One cut plane reduced to a half-space in scene space: an upward normal, its
// signed offset, and the height used to tell an upper plane from a lower one.
interface PreparedPlane {
  d0: number;
  nrm: THREE.Vector3;
  height: number;
}

// Splits a point cloud into the part the cut keeps and the part it discards.
export function splitByPlanes(
  positions: Float32Array,
  planes: CutPlane[],
  sceneScale: number,
  offset: THREE.Vector3,
): { keep: Float32Array; drop: Float32Array } {
  const prepared = preparePlanes(planes, sceneScale, offset);
  return splitPrepared(positions, prepared);
}

// Turns review planes into upward-facing half-spaces in scene space. Shared by
// the point split and the surface clip so the two cannot disagree about which
// side is kept.
export function preparePlanes(
  planes: CutPlane[],
  sceneScale: number,
  offset: THREE.Vector3,
): PreparedPlane[] {
  const UP = new THREE.Vector3(0, 1, 0);
  return planes.slice(0, MAX_PLANES).flatMap((p) => {
    // Plane data is in levelled Z-up mesh units; scene space is Y-up cm, and
    // the cloud was also recentred — `offset` carries that same translation.
    const c = pointToScene(p.centroid, sceneScale, offset);
    const nrm = dirToScene(p.normal);
    const vert = nrm.dot(UP);
    // A plane standing vertical has no above or below, so the rule is
    // undefined for it. Skipping beats guessing a side.
    if (Math.abs(vert) < 1e-3) return [];
    if (vert < 0) nrm.negate();
    return [{ d0: nrm.dot(c), nrm, height: c.dot(UP) }];
  });

}

// Splits the positions against prepared half-spaces: below a single plane, or
// between two.
function splitPrepared(
  positions: Float32Array,
  prepared: PreparedPlane[],
): { keep: Float32Array; drop: Float32Array } {
  const n = positions.length / 3;
  if (!prepared.length) return { keep: positions, drop: new Float32Array(0) };

  const keep: number[] = [];
  const drop: number[] = [];
  const v = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    v.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    let kept: boolean;
    if (prepared.length === 1) {
      kept = prepared[0].nrm.dot(v) - prepared[0].d0 <= 0;
    } else {
      // A point between the planes is below the upper and above the lower, so
      // exactly one of the two tests is true — no need to know which plane is
      // which.
      const a = prepared[0].nrm.dot(v) - prepared[0].d0 <= 0;
      const b = prepared[1].nrm.dot(v) - prepared[1].d0 <= 0;
      kept = a !== b;
    }
    const target = kept ? keep : drop;
    target.push(v.x, v.y, v.z);
  }
  return { keep: new Float32Array(keep), drop: new Float32Array(drop) };
}

/**
 * Horizontal centre of the object at mid height, in scene cm.
 *
 * The bounding-box centre is not this: on a leg the foot juts forward, so the
 * box centre sits ahead of the calf and a plane seeded there hangs off the
 * front. Averaging only the points inside a band around mid height gives the
 * limb's own axis where a cut would actually be made.
 */
function midAxis(geom: THREE.BufferGeometry): THREE.Vector2 {
  const pos = geom.getAttribute("position").array as Float32Array;
  const bb = geom.boundingBox!;
  const mid = (bb.min.y + bb.max.y) / 2;
  const half = (bb.max.y - bb.min.y) * 0.1;
  let sx = 0, sz = 0, k = 0;
  for (let i = 0; i < pos.length; i += 3) {
    if (Math.abs(pos[i + 1] - mid) > half) continue;
    sx += pos[i];
    sz += pos[i + 2];
    k++;
  }
  if (!k) return new THREE.Vector2((bb.min.x + bb.max.x) / 2, (bb.min.z + bb.max.z) / 2);
  return new THREE.Vector2(sx / k, sz / k);
}


// Renders the limb as a solid surface, clipped to the region the cut keeps.
// Used whenever the loaded geometry carries faces; a point cloud falls back to
// the Points path below. A surface is what a clinician actually places a cut
// on — judging where a plane meets skin is far harder against a scatter of
// points than against the skin itself.
function Surface({
  geometry,
  prepared,
}: {
  geometry: THREE.BufferGeometry;
  prepared: PreparedPlane[];
}) {
  const clippingPlanes = useMemo(() => {
    // three.js keeps whatever satisfies normal·p + constant >= 0, and the
    // planes here are prepared with their normals pointing up.
    if (prepared.length === 0) return [];
    if (prepared.length === 1) {
      // Keep what is BELOW the plane.
      const only = prepared[0];
      return [new THREE.Plane(only.nrm.clone().negate(), only.d0)];
    }
    // Keep what lies BETWEEN: below the upper plane and above the lower one.
    // Several clipping planes intersect, which is exactly that conjunction.
    const byHeight = [...prepared].sort((a, b) => a.height - b.height);
    const lower = byHeight[0];
    const upper = byHeight[1];
    return [
      new THREE.Plane(upper.nrm.clone().negate(), upper.d0),
      new THREE.Plane(lower.nrm.clone(), -lower.d0),
    ];
  }, [prepared]);

  return (
    <group>
      {/* The discarded remainder, faint, so the cut reads as a cut rather than
          as the limb simply ending where it does. */}
      <mesh geometry={geometry}>
        <meshStandardMaterial
          color="#b9bcc0"
          transparent
          opacity={0.16}
          depthWrite={false}
          roughness={0.9}
        />
      </mesh>
      <mesh geometry={geometry}>
        <meshStandardMaterial
          color="#3b7dd8"
          clippingPlanes={clippingPlanes}
          roughness={0.65}
          metalness={0.05}
        />
      </mesh>
    </group>
  );
}

// Renders one point cloud as a three.js points object.
function Points({
  data,
  color,
  size,
  opacity = 1,
}: {
  data: Float32Array;
  color: string;
  size: number;
  opacity?: number;
}) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(data, 3));
    return g;
  }, [data]);
  if (!data.length) return null;
  return (
    <points geometry={geom}>
      <pointsMaterial
        size={size}
        color={color}
        sizeAttenuation
        transparent={opacity < 1}
        opacity={opacity}
      />
    </points>
  );
}

/** Semi-transparent disc showing where a plane cuts. */
function PlaneWidget({
  plane,
  sceneScale,
  offset,
  radius,
  active,
}: {
  plane: CutPlane;
  sceneScale: number;
  offset: THREE.Vector3;
  radius: number;
  active: boolean;
}) {
  const { position, quaternion } = useMemo(() => {
    const pos = pointToScene(plane.centroid, sceneScale, offset);
    const nrm = dirToScene(plane.normal);
    // planeGeometry faces +Z; rotate that onto the plane normal.
    const q = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 0, 1),
      nrm,
    );
    return { position: pos, quaternion: q };
  }, [plane, sceneScale, offset]);

  return (
    <group position={position} quaternion={quaternion}>
      <mesh>
        <circleGeometry args={[radius, 64]} />
        <meshBasicMaterial
          color={active ? "#2fae6b" : "#7fae95"}
          transparent
          opacity={active ? 0.34 : 0.18}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <mesh>
        <ringGeometry args={[radius * 0.985, radius, 64]} />
        <meshBasicMaterial
          color={active ? "#2fae6b" : "#7fae95"}
          transparent
          opacity={0.95}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

/**
 * The marker line as detection found it. Fixed — this is a record of a
 * measurement, so nothing in the UI moves it.
 *
 * Drawn in yellow behind the green cut disc. Once the user drags a plane there
 * is otherwise nothing left on screen showing where the marker actually was,
 * and no way to judge how far the proposed cut has drifted from it. Slightly
 * wider than the cut disc, and drawn first, so the two do not z-fight while
 * they still coincide.
 */
function OriginWidget({
  plane,
  sceneScale,
  offset,
  radius,
}: {
  plane: CutPlane;
  sceneScale: number;
  offset: THREE.Vector3;
  radius: number;
}) {
  const placed = useMemo(() => {
    if (!plane.origin) return null;
    const pos = pointToScene(plane.origin.centroid, sceneScale, offset);
    const nrm = dirToScene(plane.origin.normal);
    const q = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 0, 1),
      nrm,
    );
    return { pos, q };
  }, [plane.origin, sceneScale, offset]);

  if (!placed) return null;
  return (
    <group position={placed.pos} quaternion={placed.q} renderOrder={-1}>
      <mesh>
        <circleGeometry args={[radius, 64]} />
        <meshBasicMaterial
          color="#e0a91b"
          transparent
          opacity={0.2}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <mesh>
        <ringGeometry args={[radius * 0.985, radius, 64]} />
        <meshBasicMaterial
          color="#e0a91b"
          transparent
          opacity={0.95}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

// Interactive 3D view of the limb with the cutting planes drawn over it.
export function CutReview({
  url,
  onLoadError,
  scale,
  planes,
  activePlaneId,
  onCounts,
  onCrossSections,
  onExtent,
}: {
  url: string;
  scale: number;
  planes: CutPlane[];
  activePlaneId: string | null;
  onCounts?: (kept: number, dropped: number) => void;
  /** Circumference at each plane, keyed by plane id, recomputed as it moves. */
  onCrossSections?: (sections: Record<string, CrossSectionResult>) => void;
  /** Vertical extent of the loaded cloud in scene cm, the whole translation
   *  the loader applied, and the object's horizontal axis at mid height —
   *  everything a manually added plane needs to land on the object. */
  onExtent?: (
    lo: number,
    hi: number,
    offset: THREE.Vector3,
    midAxis: THREE.Vector2,
  ) => void;
  /** Called with the load failure, or null once it loads. */
  onLoadError?: (message: string | null) => void;
}) {
  const { geometry, error } = usePly(url, scale);
  // This component draws inside the Canvas, so it cannot show a message itself
  // — DOM does not exist in here. Hand the failure to whoever owns the Viewport.
  useEffect(() => {
    onLoadError?.(error);
  }, [error, onLoadError]);

  const split = useMemo(() => {
    if (!geometry) return null;
    const pos = geometry.getAttribute("position").array as Float32Array;
    const off =
      (geometry.userData.sceneOffset as THREE.Vector3) ?? new THREE.Vector3();
    const r = splitByPlanes(pos, planes, scale, off);
    onCounts?.(r.keep.length / 3, r.drop.length / 3);

    // Circumference at each plane, on the same points and the same pose the
    // split just used, so the number and the picture cannot disagree. The fit
    // is `lib/crosssection.ts`, a port of the Stage 6 code, and it runs on the
    // UNCUT cloud: what is being measured is the limb's girth where the plane
    // crosses it, which does not depend on which side the cut keeps.
    const sections: Record<string, CrossSectionResult> = {};
    const floorY = geometry.boundingBox?.min.y ?? null;
    for (const p of planes.slice(0, MAX_PLANES)) {
      const c = pointToScene(p.centroid, scale, off);
      const n = dirToScene(p.normal);
      sections[p.id] = fitSlice(pos, [c.x, c.y, c.z], [n.x, n.y, n.z], floorY);
    }
    onCrossSections?.(sections);
    return r;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geometry, planes, scale]);

  useEffect(() => {
    if (!geometry) return;
    const bb = geometry.boundingBox!;
    const off = geometry.userData.sceneOffset as THREE.Vector3 | undefined;
    onExtent?.(bb.min.y, bb.max.y, off ?? new THREE.Vector3(), midAxis(geometry));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geometry]);

  if (!geometry || !split) return null;

  const bb = geometry.boundingBox!;
  const radius =
    Math.max(bb.max.x - bb.min.x, bb.max.z - bb.min.z) * 0.85 || 6;

  return (
    <Bounds fit clip observe margin={1.5}>
      <group>
        {/* Stage 5 publishes a watertight solid, and that is what the cut is
            actually applied to, so the review draws it as a surface. Jobs
            measured before the cut moved — and the bundled samples — still
            arrive as point clouds, which have no index; those fall back to the
            point split. */}
        {geometry.index ? (
          <Surface
            geometry={geometry}
            prepared={preparePlanes(
              planes,
              scale,
              (geometry.userData.sceneOffset as THREE.Vector3) ??
                new THREE.Vector3(),
            )}
          />
        ) : (
          <>
            <Points data={split.keep} color="#3b7dd8" size={0.22} />
            <Points data={split.drop} color="#b9bcc0" size={0.16} opacity={0.4} />
          </>
        )}
        {planes.map((p) => (
          <OriginWidget
            key={`o${p.id}`}
            plane={p}
            sceneScale={scale}
            offset={
              (geometry.userData.sceneOffset as THREE.Vector3) ??
              new THREE.Vector3()
            }
            radius={radius * 1.06}
          />
        ))}
        {planes.map((p) => (
          <PlaneWidget
            key={p.id}
            plane={p}
            sceneScale={scale}
            offset={
              (geometry.userData.sceneOffset as THREE.Vector3) ??
              new THREE.Vector3()
            }
            radius={radius}
            active={p.id === activePlaneId}
          />
        ))}
      </group>
    </Bounds>
  );
}
