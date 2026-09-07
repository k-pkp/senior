import type { CutPlane, SampleDataset, VolumeRow } from "./types";

/** Precomputed runs shipped with the site. These work with no backend at all —
 *  see prompt.md §4c. Only new uploads need the GPU service. */
export const SAMPLES: SampleDataset[] = [
  {
    id: "small_leg",
    label: "Lower leg",
    subject: "Human lower leg, marker band below the knee",
    nominalMl: null, // no ground truth for a limb — do not invent one
    frames: 6,
    meshes: {
      leg: "/samples/small_leg/leg_mesh.ply",
      box: "/samples/small_leg/box_mesh.ply",
      scene: "/samples/small_leg/scene_mesh.ply",
      legNoCut: "/samples/small_leg/leg_no_cut.ply",
    },
    volumesCsv: "/samples/small_leg/volumes.csv",
    cuttingLine: "/samples/small_leg/cutting_line.json",
  },
  {
    id: "est_325",
    label: "est 325",
    subject: "325 ml drink can, no marker band",
    nominalMl: 325,
    frames: 8,
    meshes: {
      leg: "/samples/est_325/leg_mesh.ply",
      box: "/samples/est_325/box_mesh.ply",
      scene: "/samples/est_325/scene_mesh.ply",
      legNoCut: "/samples/est_325/leg_no_cut.ply",
    },
    volumesCsv: "/samples/est_325/volumes.csv",
    cuttingLine: "/samples/est_325/cutting_line.json",
  },
];

/** Stage list for the processing screen. Times measured on an RTX 4060,
 *  6 photos. Do not inflate these.
 *
 * The pipeline runs in two passes with the user in between, so the same stage
 * numbers mean different things either side of the review. Naming them by what
 * they produce in THAT pass is the only honest labelling: on the first pass
 * stages 4-6 build the limb's solid and measure the reference cube, but do not
 * measure the limb: its extent is not known until the cut is confirmed. */
export const MEASURE_STAGES = [
  { n: 1, label: "VGGT inference", out: "predictions.npz", seconds: 50 },
  { n: 2, label: "Point cloud export", out: "points.ply", seconds: 2 },
  { n: 3, label: "Segment & find the cut", out: "leg.ply", seconds: 8 },
  { n: 4, label: "Surface reconstruction", out: "leg_recon.ply", seconds: 8 },
  { n: 5, label: "Watertight solid, left uncut", out: "leg_no_cut.ply", seconds: 2 },
  { n: 6, label: "Measure the reference", out: "volumes.csv", seconds: 2 },
] as const;

/** The second pass: the cut the user confirmed, then the object itself. Only
 *  stages 5 and 6 appear. The cut is a plane slice through the watertight solid
 *  Stage 5 already built, so neither the segmentation nor the surface
 *  reconstruction is repeated — an edit costs seconds. */
export const CUT_STAGES = [
  { n: 5, label: "Apply the confirmed cut", out: "leg_cut.ply", seconds: 3 },
  { n: 6, label: "Volume by surface integration", out: "volumes.csv", seconds: 2 },
] as const;

export const STAGES = MEASURE_STAGES;

/** Minimal CSV parse — volumes.csv has no quoted fields. */
export function parseVolumesCsv(text: string): VolumeRow[] {
  const [header, ...lines] = text.trim().split("\n");
  const cols = header.split(",");
  return lines
    .filter((l) => l.trim())
    .map((line) => {
      const cells = line.split(",");
      // True when the CSV carries this column.
      const has = (k: string) => cols.indexOf(k) >= 0;
      // Raw cell text for this column on the current row.
      const get = (k: string) => cells[cols.indexOf(k)];
      // Cell parsed as a number, falling back to 0 when it is not finite.
      const num = (k: string) => {
        const v = parseFloat(get(k));
        return Number.isFinite(v) ? v : 0;
      };
      // Two Stage 6 implementations are in circulation and they name these
      // differently. The parked one reports an ORIENTED box (obb_*/height_cm),
      // main's reports an AXIS-ALIGNED one (ext_*/size_*_cm). Normalise to one
      // shape here so nothing downstream has to know, but remember which was
      // read — the two are not interchangeable for deriving scale, since an
      // AABB around a tilted cube measures its diagonal.
      const aabb = !has("obb_b");
      return {
        name: get("name"),
        is_ref: get("is_ref") === "True",
        volume: num("volume"),
        method: get("method"),
        // obb_a is the vertical axis; main's vertical is ext_z, the scene
        // having been levelled so Z is up.
        obb_a: aabb ? num("ext_z") : num("obb_a"),
        obb_b: aabb ? num("ext_x") : num("obb_b"),
        obb_c: aabb ? num("ext_y") : num("obb_c"),
        aabb,
        voxel: has("voxel") && get("voxel") ? num("voxel") : null,
        real_vol_cm3: num("real_vol_cm3"),
        real_vol_L: num("real_vol_L"),
        height_cm: aabb ? num("size_z_cm") : num("height_cm"),
        width_cm: aabb ? num("size_x_cm") : num("width_cm"),
        depth_cm: aabb ? num("size_y_cm") : num("depth_cm"),
      } satisfies VolumeRow;
    });
}

// Fetches and parses a run's volumes.csv.
export async function loadVolumes(url: string): Promise<VolumeRow[]> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`volumes.csv ${res.status}`);
  return parseVolumesCsv(await res.text());
}

export const REFERENCE_CM = 14.0;

/** cm per mesh unit — derived the way the Stage 6 that produced this CSV did it.
 *
 * Mirroring matters more than picking the better method. If the viewer scaled
 * geometry differently from the stage that computed the volumes, an object would
 * appear at a size that contradicts its own printed measurement, and there would
 * be no way to tell which was wrong.
 *
 * Returns null when no scale can be derived, so the caller can say so. It used
 * to return `14 / 0` here and hand Infinity to the loader, which multiplied every
 * vertex by it and produced a scene of NaNs — a blank viewport with no error,
 * indistinguishable from a browser that cannot draw at all.
 */
export function linearScale(rows: VolumeRow[]): number | null {
  const ref = rows.find((r) => r.is_ref);
  if (!ref) return null;

  // Preferred: the reference's measured EDGE, horizontals only. obb_a is the
  // vertical axis, which the floor truncates — averaging it in drags the
  // estimate small and inflates everything measured against it.
  let scale = REFERENCE_CM / ((ref.obb_b + ref.obb_c) / 2);

  if (ref.aabb) {
    // main's Stage 6. Its extents are axis-aligned, so on a tilted cube they
    // read the diagonal — using them here gives 44.0 cm/unit against a true
    // 59.8, a 26% error. Its own derivation is the volume ratio, so use that
    // instead and stay consistent with the numbers it printed. The cube root
    // treats V^(1/3) as an edge length, which only holds for a perfect cube:
    // measured +2% against the fitted-face answer on inputs/small_leg.
    scale = Math.cbrt(ref.real_vol_cm3 / ref.volume);
  }
  return Number.isFinite(scale) && scale > 0 ? scale : null;
}

let planeSeq = 0;
// Returns a fresh unique id for a newly added plane.
export function newPlaneId() {
  return `p${++planeSeq}`;
}

// Fetches the detected cutting planes, returning an empty list when there are none yet.
export async function loadCutPlanes(url: string): Promise<CutPlane[]> {
  try {
    const res = await fetch(url);
    if (!res.ok) return [];
    const json = await res.json();
    // "candidates" is every band detection validated; "markers" is the subset
    // the run happened to cut on, which --cut-mode upper trims to one even on a
    // two-band capture. The reviewer is choosing what to cut, so they get the
    // full set. Older files carry only "markers" — fall back to it.
    const raw = json.candidates?.length ? json.candidates : (json.markers ?? []);
    // Lowest first, so a caller picking "the outermost two" can take the ends.
    // The pipeline already writes candidates in this order; sorting here means
    // an older file, or a hand-edited one, cannot quietly break that.
    return raw
      .slice()
      .sort((a: any, b: any) => a.centroid[2] - b.centroid[2])
      .map((m: any) => ({
        id: newPlaneId(),
        centroid: m.centroid as [number, number, number],
        normal: m.normal as [number, number, number],
        npts: m.npts ?? 0,
        source: "detected" as const,
        origin: {
          centroid: m.centroid as [number, number, number],
          normal: m.normal as [number, number, number],
        },
      }));
  } catch {
    return [];
  }
}
