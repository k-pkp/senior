"use client";

import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls } from "@react-three/drei";
import { Suspense, useEffect, useState, type ReactNode } from "react";

/** Whether this browser can give us a 3D context at all.
 *
 * Worth testing rather than assuming: an embedded webview — VS Code's Simple
 * Browser is the one that catches people here — serves the page, the panels and
 * the controls perfectly while silently having no WebGL, so the only thing
 * missing is the part the screen exists for. A blank canvas is indistinguishable
 * from a failed load unless we say which it is. */
function webglAvailable(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

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
  error = null,
}: {
  children: ReactNode;
  cameraPosition?: [number, number, number];
  gridSize?: number;
  showGrid?: boolean;
  className?: string;
  /** Shown instead of the scene when the geometry could not be loaded. */
  error?: string | null;
}) {
  // Probed after mount, never during render: the server has no document, and a
  // value that differs between server and client would break hydration.
  const [noWebgl, setNoWebgl] = useState(false);
  useEffect(() => setNoWebgl(!webglAvailable()), []);

  const problem = noWebgl
    ? {
        title: "This browser cannot draw 3D",
        detail:
          "No WebGL context is available. If you are viewing inside an editor " +
          "preview pane, open the page in a normal browser window instead.",
      }
    : error
      ? {
          title: "Nothing to draw",
          detail: error,
        }
      : null;

  if (problem) {
    return (
      <div
        className={className}
        style={{
          position: "relative",
          width: "100%",
          height: "100%",
          background: "var(--soft)",
          borderRadius: "var(--radius)",
          display: "grid",
          placeItems: "center",
          padding: 24,
          textAlign: "center",
        }}
      >
        <div style={{ maxWidth: 380 }}>
          <div style={{ font: "500 13.5px/1.4 var(--sans)" }}>
            {problem.title}
          </div>
          <div
            style={{
              font: "400 12.5px/1.6 var(--sans)",
              color: "var(--muted)",
              marginTop: 8,
            }}
          >
            {problem.detail}
          </div>
        </div>
      </div>
    );
  }

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
        // localClippingEnabled lets a material carry its own clipping planes,
        // which is how the review shows the kept side of the cut as a solid.
        gl={{ antialias: true, localClippingEnabled: true }}
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
