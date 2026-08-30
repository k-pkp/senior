"use client";

/**
 * Place the cut on a photograph instead of on the reconstruction.
 *
 * The marker band is visible in the photo and only inferred in the mesh, so
 * clicking the band is a more direct action than judging where a translucent
 * disc meets a surface. The frame shown is the one VGGT actually consumed —
 * Stage 0's 518px crop — and the camera is built from that frame's own
 * extrinsic and intrinsic, so the mesh drawn over it lines up to under a pixel.
 *
 * What a click contributes, and what it does not:
 *
 *   - the image gives the HEIGHT. The click casts a ray, the ray hits the
 *     limb, and the hit point fixes where along the limb the cut falls.
 *   - the limb gives the ORIENTATION. The plane is vertical in levelled space,
 *     the same convention a hand-placed plane already uses.
 *
 * Deriving the orientation from the image too would be a mistake worth naming:
 * a line drawn across the photo back-projects to a plane through the camera
 * centre, so the cut would tilt according to where the photographer happened to
 * stand, and two people cutting the same band on different frames would get
 * different volumes.
 */
import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";

import {
  CameraData,
  LevellingData,
  cameraForFrame,
  rayThroughImagePoint,
} from "@/lib/imagecamera";
import { CutPlane } from "@/lib/types";
import { newPlaneId } from "@/lib/data";
import { usePly } from "./usePly";

/** Side of the square frame VGGT consumes. Stage 0 emits exactly this. */
const FRAME_SIZE = 518;

/**
 * Converts a point in scene space back to the levelled mesh space the pipeline
 * stores planes in.
 *
 * `transformToScene` rotates mesh (x, y, z) to scene (x, z, -y), then scales,
 * then translates. This undoes all three, in reverse.
 */
function sceneToMesh(
  point: THREE.Vector3,
  scale: number,
  offset: THREE.Vector3,
): [number, number, number] {
  return [
    (point.x - offset.x) / scale,
    -(point.z - offset.z) / scale,
    (point.y - offset.y) / scale,
  ];
}

// The image-cut view: one of VGGT's own frames with the limb drawn over it,
// where clicking the marker band places the cut at that point.
export function ImageCut({
  url,
  frameUrls,
  cameras,
  levelling,
  scale,
  disabled,
  onPlacePlane,
}: {
  url: string;
  frameUrls: string[];
  cameras: CameraData | null;
  levelling: LevellingData | null;
  scale: number;
  disabled?: boolean;
  onPlacePlane: (plane: CutPlane) => void;
}) {
  // The same solid the 3D view draws. Loading it here rather than receiving it
  // keeps the two views independent; the browser serves the second request from
  // cache.
  const { geometry } = usePly(url, scale);
  const [frameIndex, setFrameIndex] = useState(0);
  const [lastHit, setLastHit] = useState<{ x: number; y: number } | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  // A plain three.js mesh purely to raycast against. It never renders, so it
  // does not need a material with any particular appearance.
  const pickTarget = useMemo(() => {
    if (!geometry) return null;
    return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  }, [geometry]);

  const placement = useMemo(() => {
    const offset =
      (geometry?.userData.sceneOffset as THREE.Vector3) ?? new THREE.Vector3();
    return { scale, offset };
  }, [geometry, scale]);

  const frameCamera = useMemo(() => {
    if (!cameras || !levelling) return null;
    return cameraForFrame(cameras, levelling, placement, frameIndex, FRAME_SIZE);
  }, [cameras, levelling, placement, frameIndex]);

  useEffect(() => {
    setLastHit(null);
    setMessage(null);
  }, [frameIndex]);

  /** Casts the clicked pixel at the limb and turns the hit into a cut plane. */
  function handleClick(event: React.MouseEvent<HTMLDivElement>) {
    if (disabled || !frameCamera || !pickTarget) return;
    const box = event.currentTarget.getBoundingClientRect();
    const fractionX = (event.clientX - box.left) / box.width;
    const fractionY = (event.clientY - box.top) / box.height;

    const ray = rayThroughImagePoint(frameCamera, fractionX, fractionY);
    const raycaster = new THREE.Raycaster(ray.origin, ray.direction);
    const hits = raycaster.intersectObject(pickTarget, false);
    if (!hits.length) {
      setMessage("That ray missed the limb — click on the limb itself.");
      setLastHit(null);
      return;
    }

    const centroid = sceneToMesh(hits[0].point, placement.scale, placement.offset);
    onPlacePlane({
      id: newPlaneId(),
      centroid,
      // Vertical in levelled space, matching a hand-placed plane. The image
      // fixes where the cut falls, not how it is angled.
      normal: [0, 0, 1],
      npts: 0,
      source: "user",
    });
    setLastHit({ x: fractionX, y: fractionY });
    setMessage(null);
  }

  if (!frameUrls.length) {
    return (
      <div style={{ font: "13px/1.5 var(--sans)", color: "var(--muted)" }}>
        No frames available for this run.
      </div>
    );
  }

  const ready = Boolean(frameCamera && pickTarget);

  return (
    <div style={{ display: "grid", gap: 10 }}>
      <div
        onClick={handleClick}
        style={{
          position: "relative",
          width: "100%",
          aspectRatio: "1 / 1",
          borderRadius: "var(--radius)",
          overflow: "hidden",
          backgroundImage: `url(${frameUrls[frameIndex]})`,
          backgroundSize: "cover",
          cursor: ready && !disabled ? "crosshair" : "default",
        }}
      >
        {lastHit ? (
          <div
            style={{
              position: "absolute",
              left: `${lastHit.x * 100}%`,
              top: `${lastHit.y * 100}%`,
              width: 14,
              height: 14,
              marginLeft: -7,
              marginTop: -7,
              borderRadius: "50%",
              border: "2px solid #3b7dd8",
              background: "rgba(59,125,216,0.35)",
            }}
          />
        ) : null}
      </div>

      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {frameUrls.map((url, index) => (
          <button
            key={url}
            onClick={() => setFrameIndex(index)}
            style={{
              width: 46,
              height: 46,
              padding: 0,
              borderRadius: 6,
              cursor: "pointer",
              backgroundImage: `url(${url})`,
              backgroundSize: "cover",
              border:
                index === frameIndex
                  ? "2px solid #3b7dd8"
                  : "1px solid var(--line)",
            }}
            aria-label={`Frame ${index + 1}`}
          />
        ))}
      </div>

      <div style={{ font: "12px/1.5 var(--sans)", color: "var(--muted)" }}>
        {message
          ? message
          : ready
            ? "Click the marker band in the photo to place the cut there."
            : "Loading the camera for this frame…"}
      </div>
    </div>
  );
}
