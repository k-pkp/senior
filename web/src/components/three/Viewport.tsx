"use client";

import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import { Suspense, type ReactNode } from "react";

/**
 * Shared 3D stage.
 *
 * Coordinate convention: the pipeline levels the scene so Z is up, three.js is
 * Y-up. Every loaded geometry is rotated -90 deg about X (see plyToScene in
 * usePly). Because Stage 3 levels against the *detected* ground plane, the
 * object's base genuinely sits at y=0 — so the grid is a real floor, not
 * decoration, and objects stand on it.
 *
 * Scene units are centimetres: geometry is scaled by `linear_scale` at load, so
 * one grid cell is 1 cm and dimension labels need no conversion.
 */
export function Viewport({
  children,
  cameraPosition = [34, 26, 34],
  gridSize = 60,
  showGrid = true,
  className,
}: {
  children: ReactNode;
  cameraPosition?: [number, number, number];
  gridSize?: number;
  showGrid?: boolean;
  className?: string;
}) {
  return (
    <div
      className={className}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        background: "var(--soft)",
        borderRadius: "var(--radius)",
        overflow: "hidden",
      }}
    >
      <Canvas
        dpr={[1, 2]}
        camera={{ position: cameraPosition, fov: 40, near: 0.1, far: 2000 }}
        gl={{ antialias: true }}
      >
        <hemisphereLight intensity={0.55} groundColor="#404040" />
        <directionalLight position={[18, 30, 14]} intensity={1.5} />
        <directionalLight position={[-16, 12, -12]} intensity={0.45} />

        <Suspense fallback={null}>{children}</Suspense>

        {showGrid && (
          <Grid
            args={[gridSize, gridSize]}
            cellSize={1}
            cellThickness={0.5}
            sectionSize={10}
            sectionThickness={1}
            infiniteGrid
            fadeDistance={gridSize * 2.2}
            fadeStrength={1.5}
            cellColor="#8a8a8a"
            sectionColor="#5f5f5f"
            followCamera={false}
          />
        )}

        <OrbitControls
          makeDefault
          enableDamping
          dampingFactor={0.08}
          minDistance={4}
          maxDistance={400}
          // Keep the camera above the floor; looking up through the grid at a
          // measurement reads as a bug, not a feature.
          maxPolarAngle={Math.PI / 2 - 0.02}
          target={[0, 6, 0]}
        />
      </Canvas>
    </div>
  );
}
