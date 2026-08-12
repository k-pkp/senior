"use client";

import { useEffect, useState } from "react";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

/**
 * Load a PLY and place it in scene space.
 *
 * Two corrections happen here, and both matter:
 *
 *  1. Z-up -> Y-up. The pipeline levels the scene so Z is vertical; three.js is
 *     Y-up. Without the -90 deg X rotation the limb lies on its side and the
 *     grid does not meet its base.
 *
 *  2. mesh units -> centimetres, via `scale` (linear_scale from volumes.csv).
 *     Doing it once at load means every downstream number and the grid are
 *     already in real units.
 *
 * The object is then centred on X/Z and dropped so its lowest point sits at
 * y=0, i.e. standing on the grid rather than floating through it.
 */
export function transformToScene(geom: THREE.BufferGeometry, scale: number) {
  geom.rotateX(-Math.PI / 2); // Z-up -> Y-up
  geom.scale(scale, scale, scale); // mesh units -> cm
  geom.computeBoundingBox();
  const bb = geom.boundingBox!;
  const cx = (bb.min.x + bb.max.x) / 2;
  const cz = (bb.min.z + bb.max.z) / 2;
  const offset = new THREE.Vector3(-cx, -bb.min.y, -cz);
  geom.translate(offset.x, offset.y, offset.z);
  geom.computeVertexNormals(); // exported PLYs do not always carry normals
  geom.computeBoundingBox();

  // Anything positioned in the same space as this geometry — the cut planes,
  // most obviously — has to receive the identical transform. Without it the
  // plane floats away from the cloud by exactly this offset.
  geom.userData.sceneOffset = offset;
  geom.userData.sceneScale = scale;
  return geom;
}

/** Apply the same rotate → scale → translate a geometry received. */
export function pointToScene(
  p: readonly [number, number, number],
  scale: number,
  offset: THREE.Vector3,
) {
  return new THREE.Vector3(p[0], p[1], p[2])
    .applyAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2)
    .multiplyScalar(scale)
    .add(offset);
}

/** Directions rotate but never translate or scale. */
export function dirToScene(d: readonly [number, number, number]) {
  return new THREE.Vector3(d[0], d[1], d[2])
    .applyAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2)
    .normalize();
}

interface PlyState {
  geometry: THREE.BufferGeometry | null;
  error: string | null;
  loading: boolean;
}

export function usePly(url: string | null, scale = 1): PlyState {
  const [state, setState] = useState<PlyState>({
    geometry: null,
    error: null,
    loading: !!url,
  });

  useEffect(() => {
    if (!url) {
      setState({ geometry: null, error: null, loading: false });
      return;
    }
    let cancelled = false;
    let created: THREE.BufferGeometry | null = null;
    setState({ geometry: null, error: null, loading: true });

    new PLYLoader().load(
      url,
      (geom) => {
        if (cancelled) {
          geom.dispose();
          return;
        }
        created = transformToScene(geom, scale);
        setState({ geometry: created, error: null, loading: false });
      },
      undefined,
      () => {
        if (!cancelled)
          setState({
            geometry: null,
            error: `Could not load ${url.split("/").pop()}`,
            loading: false,
          });
      },
    );

    return () => {
      cancelled = true;
      created?.dispose();
    };
  }, [url, scale]);

  return state;
}

/** Points of a geometry as a flat Float32Array, for the client-side cut test. */
export function positionsOf(geom: THREE.BufferGeometry): Float32Array {
  return geom.getAttribute("position").array as Float32Array;
}
