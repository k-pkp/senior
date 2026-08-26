"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { Button, Caveat, Label, Panel, Slider } from "@/components/ui/primitives";
import { linearScale, loadCutPlanes, loadVolumes, newPlaneId } from "@/lib/data";
import { MAX_PLANES } from "@/components/three/CutReview";
import type { CutPlane, SampleDataset, VolumeRow } from "@/lib/types";

const Viewport = dynamic(
  () => import("@/components/three/Viewport").then((m) => m.Viewport),
  { ssr: false },
);
const CutReview = dynamic(
  () => import("@/components/three/CutReview").then((m) => m.CutReview),
  { ssr: false },
);

/** Height slider works in scene cm; tilt/direction rebuild the normal. */
function planeFromControls(
  base: CutPlane,
  heightCm: number,
  tiltDeg: number,
  dirDeg: number,
  scale: number,
  offsetY: number,
): CutPlane {
  const tilt = THREE.MathUtils.degToRad(tiltDeg);
  const dir = THREE.MathUtils.degToRad(dirDeg);
  // Start vertical (Z up in pipeline space), tilt away from it, spin the tilt
  // direction around the vertical axis.
  const n = new THREE.Vector3(
    Math.sin(tilt) * Math.cos(dir),
    Math.sin(tilt) * Math.sin(dir),
    Math.cos(tilt),
  ).normalize();
  // rotateX(-90) sends mesh Z to scene Y, and the loader then translated by
  // offsetY — so scene height = meshZ * scale + offsetY. Invert that.
  return {
    ...base,
    centroid: [base.centroid[0], base.centroid[1], (heightCm - offsetY) / scale],
    normal: [n.x, n.y, n.z],
  };
}

interface Controls {
  height: number;
  tilt: number;
  dir: number;
}

/** Slider values that reproduce `normal` exactly through planeFromControls.
 *
 * planeFromControls always builds an upward normal (`cos(tilt)` with tilt in
 * ±35 deg), so a detected normal pointing down has to be folded up first. A
 * plane is unchanged by negating its whole normal, but NOT by flipping z
 * alone — reading tilt as `acos(|n.z|)` while keeping the original x/y did
 * exactly that, so the plane visibly reversed its tilt the moment any slider
 * was touched.
 */
function controlsFromNormal(
  normal: readonly [number, number, number],
  heightCm: number,
): Controls {
  const n = new THREE.Vector3(...normal).normalize();
  if (n.z < 0) n.negate();
  return {
    height: heightCm,
    tilt: THREE.MathUtils.radToDeg(Math.acos(Math.min(1, n.z))),
    dir: THREE.MathUtils.radToDeg(Math.atan2(n.y, n.x)),
  };
}

export function Review({
  dataset,
  live,
  onConfirm,
  onBack,
}: {
  dataset: SampleDataset;
  /** True when this is a live job the service can re-cut. False for a shipped
   *  sample, whose meshes are static files — the plane can still be moved to
   *  see what a different cut would keep, but nothing can be re-measured. */
  live: boolean;
  /** Handed the planes as they stand. For a live job that means re-running
   *  stages 3-6 with them; for a shipped sample there is nothing to re-run, so
   *  it just moves on. Review itself does not need to know which. */
  onConfirm: (planes: CutPlane[]) => void;
  onBack: () => void;
}) {
  const [rows, setRows] = useState<VolumeRow[] | null>(null);
  const [planes, setPlanes] = useState<CutPlane[]>([]);
  const [controls, setControls] = useState<Record<string, Controls>>({});
  const [active, setActive] = useState<string | null>(null);
  const [counts, setCounts] = useState({ kept: 0, dropped: 0 });
  // Height slider spans the object's own vertical extent, reported by the
  // viewer once the cloud is loaded. Hardcoded bounds do not survive a
  // different subject.
  const [zRange, setZRange] = useState<[number, number]>([0, 30]);
  // The loader's full translation. offset.y converts slider cm <-> mesh Z;
  // offset.x/z are what put a manually added plane on the object's axis.
  const [offset, setOffset] = useState(new THREE.Vector3());
  const offsetY = offset.y;
  // Where the object actually is horizontally at mid height, in scene cm.
  const [midAxis, setMidAxis] = useState(new THREE.Vector2());
  // Plane seeding runs before the geometry reports its offset; a ref keeps the
  // latest value readable from that effect without re-triggering it.
  const [calibrated, setCalibrated] = useState(false);
  // A mesh that fails to load leaves an empty viewport, which looks identical to
  // a browser that cannot draw at all and to a scene with nothing in it. Carry
  // the reason out so the viewport can say which.
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    loadVolumes(dataset.volumesCsv).then(setRows).catch(() => setRows([]));
  }, [dataset]);

  const scaleRaw = useMemo(() => (rows ? linearScale(rows) : null), [rows]);
  // 1 is the "not known yet" placeholder the calibration effect already watches
  // for; null means it can never be known for this run, which is different.
  const scale = scaleRaw ?? 1;
  const scaleError =
    rows && scaleRaw === null
      ? "This run was measured by a Stage 6 that does not report the reference "
        + "cube's edge length, so the scene cannot be drawn to scale."
      : null;

  useEffect(() => {
    loadCutPlanes(dataset.cuttingLine).then((detected) => {
      // A plane appears only when a marker was actually detected. Seeding a
      // fake one would imply a detection that never happened, and the user
      // would be adjusting an invention rather than a measurement. With no
      // marker the user adds a plane deliberately, or continues uncut.
      setPlanes(detected.slice(0, MAX_PLANES));
      setActive(detected[0]?.id ?? null);
      setCalibrated(false);   // control values need offsetY, which arrives with the geometry
    });
  }, [dataset, scale]);

  // Slider readouts depend on the loader's translation, which is only known
  // once the cloud has loaded. Deriving them at seed time gave a height of
  // -17.53 cm — the plane was drawn correctly, the number was not.
  useEffect(() => {
    // offsetY starts at 0 and only becomes real once the cloud has loaded and
    // reported its bounds. Calibrating before then bakes in a wrong height and
    // then marks itself done, which is what left Height reading -17.53 cm.
    if (calibrated || planes.length === 0 || scale === 1 || offsetY === 0) return;
    const c: Record<string, Controls> = {};
    for (const p of planes) {
      c[p.id] = controlsFromNormal(p.normal, p.centroid[2] * scale + offsetY);
    }
    setControls(c);
    setCalibrated(true);
  }, [planes, offsetY, scale, calibrated]);

  function update(id: string, patch: Partial<Controls>) {
    setControls((prev) => {
      const next = { ...prev, [id]: { ...prev[id], ...patch } };
      setPlanes((ps) =>
        ps.map((p) =>
          p.id === id
            ? planeFromControls(
                p,
                next[id].height,
                next[id].tilt,
                next[id].dir,
                scale,
                offsetY,
              )
            : p,
        ),
      );
      return next;
    });
  }

  function addPlane() {
    if (planes.length >= MAX_PLANES) return;
    const id = newPlaneId();
    const midY = (zRange[0] + zRange[1]) / 2;
    // A detected plane carries the marker's own centroid, so it sits on the
    // object. [0,0,z] is the *mesh* origin, which the loader's recentring moved
    // away from the object — that is why a manual plane appeared off to one
    // side. Put it on the object's axis instead, converted back to mesh space:
    // rotateX(-90) sends mesh (x,y,z) to scene (x,z,-y), so scene.x =
    // meshX*s + off.x and scene.z = -meshY*s + off.z. Invert both.
    const p: CutPlane = {
      id,
      centroid: [
        (midAxis.x - offset.x) / scale,
        -(midAxis.y - offset.z) / scale,
        (midY - offsetY) / scale,
      ],
      normal: [0, 0, 1],
      npts: 0,
      source: "user",
    };
    setPlanes((ps) => [...ps, p]);
    setControls((c) => ({ ...c, [id]: { height: midY, tilt: 0, dir: 0 } }));
    setActive(id);
  }

  /** Put a dragged plane back exactly where detection had it. */
  function resetPlane(id: string) {
    const p = planes.find((x) => x.id === id);
    if (!p?.origin) return;
    setPlanes((ps) =>
      ps.map((x) =>
        x.id === id
          ? { ...x, centroid: p.origin!.centroid, normal: p.origin!.normal }
          : x,
      ),
    );
    setControls((c) => ({
      ...c,
      [id]: controlsFromNormal(
        p.origin!.normal,
        p.origin!.centroid[2] * scale + offsetY,
      ),
    }));
  }

  function removePlane(id: string) {
    setPlanes((ps) => ps.filter((p) => p.id !== id));
    setActive((a) => (a === id ? null : a));
  }

  return (
    <div className="fadein" style={{ display: "grid", gap: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <Button variant="ghost" onClick={onBack}>
          ← Back
        </Button>
        <div style={{ font: "500 15px/1 var(--sans)" }}>Review the cut</div>
        <div
          style={{
            marginLeft: "auto",
            font: "400 12px/1 var(--mono)",
            color: "var(--muted)",
          }}
        >
          {planes.length === 0
            ? `${counts.kept.toLocaleString()} points · uncut`
            : `${counts.kept.toLocaleString()} kept · ${counts.dropped.toLocaleString()} discarded`}
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0,1fr) 320px",
          gap: 16,
          alignItems: "start",
        }}
        className="review-grid"
      >
        <Panel pad={0} style={{ overflow: "hidden" }}>
          <div style={{ height: "clamp(380px, 58vh, 620px)" }}>
            <Viewport error={scaleError ?? loadError}>
              {rows && (
                <CutReview
                  onLoadError={setLoadError}
                  url={dataset.meshes.legNoCut}
                  scale={scale}
                  planes={planes}
                  activePlaneId={active}
                  onCounts={(kept, dropped) => setCounts({ kept, dropped })}
                  onExtent={(lo, hi, off, axis) => {
                    setZRange([lo, hi]);
                    setOffset(off);
                    setMidAxis(axis);
                  }}
                />
              )}
            </Viewport>
          </div>
          <div
            style={{
              padding: 12,
              borderTop: "1px solid var(--line)",
              font: "400 11.5px/1.5 var(--sans)",
              color: "var(--muted)",
            }}
          >
            Blue points are kept, grey are discarded.{" "}
            The <span style={{ color: "#b8860b" }}>yellow plane</span> is the
            marker line as detected — it is a record of a measurement and cannot
            be moved. The green disc is the cut you are proposing. 1 grid square
            = 1 cm.
          </div>
        </Panel>

        <div style={{ display: "grid", gap: 12 }}>
          {planes.map((p, i) => (
            <Panel
              key={p.id}
              style={{
                outline:
                  p.id === active ? "2px solid var(--accent)" : "none",
                outlineOffset: -1,
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 12,
                }}
                onClick={() => setActive(p.id)}
              >
                <Label>
                  Plane {i + 1}
                  {p.source === "detected" ? ` · ${p.npts} marker pts` : " · manual"}
                </Label>
                <Button
                  variant="quiet"
                  onClick={() => removePlane(p.id)}
                  style={{ padding: "2px 6px", fontSize: 11 }}
                >
                  remove
                </Button>
              </div>
              {p.origin && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "baseline",
                    gap: 8,
                    marginBottom: 10,
                    font: "400 11.5px/1.5 var(--sans)",
                    color: "var(--muted)",
                  }}
                >
                  <span
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: 8,
                      background: "#e0a91b",
                      flex: "0 0 auto",
                    }}
                  />
                  <span>
                    marker detected at{" "}
                    <span style={{ font: "400 11.5px/1.5 var(--mono)" }}>
                      {(p.origin.centroid[2] * scale + offsetY).toFixed(2)} cm
                    </span>
                  </span>
                  <Button
                    variant="quiet"
                    onClick={() => resetPlane(p.id)}
                    style={{ marginLeft: "auto", padding: "2px 6px", fontSize: 11 }}
                  >
                    relocate to marker
                  </Button>
                </div>
              )}
              <Slider
                label="Height"
                value={controls[p.id]?.height ?? 0}
                // Start 1 cm above the object's lowest point. At the very
                // bottom a single plane keeps nothing below it, so the slider
                // would have a dead end that empties the selection; one
                // centimetre of clearance always leaves points on the kept
                // side.
                min={Math.floor(zRange[0]) + 1}
                max={Math.ceil(zRange[1])}
                step={0.1}
                suffix=" cm"
                onChange={(v) => update(p.id, { height: v })}
              />
              <Slider
                label="Tilt"
                value={controls[p.id]?.tilt ?? 0}
                min={-35}
                max={35}
                suffix="°"
                onChange={(v) => update(p.id, { tilt: v })}
              />
              <Slider
                label="Tilt direction"
                value={controls[p.id]?.dir ?? 0}
                min={-180}
                max={180}
                step={5}
                suffix="°"
                onChange={(v) => update(p.id, { dir: v })}
              />
            </Panel>
          ))}

          {planes.length === 0 && (
            <Panel>
              <Label>No marker detected</Label>
              <p
                style={{
                  font: "400 12.5px/1.6 var(--sans)",
                  color: "var(--muted)",
                  margin: "9px 0 0",
                }}
              >
                No coloured marker band was found on this object, so nothing is
                being cut — the whole reconstruction will be measured. Add a
                plane below if you want to cut it manually.
              </p>
            </Panel>
          )}

          {planes.length < MAX_PLANES && (
            <Button
              variant="ghost"
              onClick={addPlane}
              style={{ borderStyle: "dashed", padding: 11, width: "100%" }}
            >
              + Add a cutting plane
            </Button>
          )}

          {planes.length >= MAX_PLANES && (
            <Caveat>
              Two planes keep the region <strong>between</strong> them, which
              extracts a segment. One plane keeps everything{" "}
              <strong>below</strong> it. Two is the limit — a third cut can only
              contradict one of these.
            </Caveat>
          )}

          <Button
            variant="primary"
            onClick={() => onConfirm(planes)}
            style={{ padding: 13, width: "100%" }}
          >
            {!live
              ? "Back to result"
              : planes.length === 0
                ? "Measure without cutting"
                : "Confirm cut & compute volume"}
          </Button>

          {!live && (
            <Caveat>
              This is a precomputed sample — the planes above re-split the
              points here in the browser, but there is nothing to re-run
              against, so the volume on the Result screen stays as it was
              measured. Upload your own photos to get a cut you can change: the
              service re-runs stages 3-6 and returns a new volume in about
              thirty seconds.
            </Caveat>
          )}
        </div>
      </div>

      <style>{`
        @media (max-width: 900px) {
          .review-grid { grid-template-columns: minmax(0,1fr) !important; }
        }
      `}</style>
    </div>
  );
}
