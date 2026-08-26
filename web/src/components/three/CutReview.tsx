"use client";

import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { Bounds } from "@react-three/drei";
import type { CutPlane } from "@/lib/types";
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
export function splitByPlanes(
  positions: Float32Array,
  planes: CutPlane[],
  sceneScale: number,
  offset: THREE.Vector3,
): { keep: Float32Array; drop: Float32Array } {
  const n = positions.length / 3;
  const UP = new THREE.Vector3(0, 1, 0);

  const prepared = planes.slice(0, MAX_PLANES).flatMap((p) => {
    // Plane data is in levelled Z-up mesh units; scene space is Y-up cm, and
    // the cloud was also recentred — `offset` carries that same translation.
    const c = pointToScene(p.centroid, sceneScale, offset);
    const nrm = dirToScene(p.normal);
    const vert = nrm.dot(UP);
    // A plane standing vertical has no above or below, so the rule is
    // undefined for it. Skipping beats guessing a side.
    if (Math.abs(vert) < 1e-3) return [];
    if (vert < 0) nrm.negate();
    return [{ d0: nrm.dot(c), nrm }];
  });

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

export function CutReview({
  url,
  onLoadError,
  scale,
  planes,
  activePlaneId,
  onCounts,
  onExtent,
}: {
  url: string;
  scale: number;
  planes: CutPlane[];
  activePlaneId: string | null;
  onCounts?: (kept: number, dropped: number) => void;
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
        <Points data={split.keep} color="#3b7dd8" size={0.22} />
        <Points data={split.drop} color="#b9bcc0" size={0.16} opacity={0.4} />
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
