"use client";

import { useEffect } from "react";
import { Bounds } from "@react-three/drei";
import * as THREE from "three";
import { usePly } from "./usePly";

/** A reconstructed mesh, vertex-coloured, auto-framed. */
export function MeshView({
  url,
  scale,
  color,
  wireframe = false,
  opacity = 1,
  onLoadError,
}: {
  url: string;
  scale: number;
  /** Overrides vertex colours when set. */
  color?: string;
  wireframe?: boolean;
  opacity?: number;
  /** Called with the load failure, or null once it loads. */
  onLoadError?: (message: string | null) => void;
}) {
  const { geometry, error } = usePly(url, scale);
  // This component draws inside the Canvas, so it cannot show a message itself
  // — DOM does not exist in here. Hand the failure to whoever owns the Viewport.
  useEffect(() => {
    onLoadError?.(error);
  }, [error, onLoadError]);
  if (!geometry) return null;

  // The exported PLYs all carry vertex colours; use them unless overridden.
  const hasColors = !!geometry.getAttribute("color");

  return (
    <Bounds fit clip observe margin={1.35}>
      <mesh geometry={geometry} castShadow receiveShadow>
        <meshStandardMaterial
          vertexColors={hasColors && !color}
          color={color ?? "#ffffff"}
          roughness={0.72}
          metalness={0.02}
          wireframe={wireframe}
          transparent={opacity < 1}
          opacity={opacity}
          side={THREE.DoubleSide}
        />
      </mesh>
    </Bounds>
  );
}
