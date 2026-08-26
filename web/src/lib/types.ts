/** Shared types. Mirrors what the pipeline actually writes — see pipeline.md. */

/** One cutting plane. Matches an entry in debug/cutting_line.json.
 *
 * IMPORTANT: the pipeline currently writes these in *original VGGT space*
 * (pre-levelling), while leg_no_cut.ply is written *after* levelling. They do
 * not share a frame. Until `clean.py` Phase C exports R_total-transformed
 * planes, the UI works in levelled space and fixtures carry levelled planes.
 */
export interface CutPlane {
  id: string;
  /** Point on the plane, levelled space (Z up), mesh units. */
  centroid: [number, number, number];
  /** Unit normal, levelled space. */
  normal: [number, number, number];
  /** Marker points that supported this plane. 0 for user-created. */
  npts: number;
  source: "detected" | "user";
  /** Where detection originally put this plane, before the user moved it.
   *  Absent on user-created planes. Kept so the review screen can show what
   *  was actually measured next to what the user is now proposing — without
   *  it, a dragged plane erases all trace of the marker it came from. */
  origin?: {
    centroid: [number, number, number];
    normal: [number, number, number];
  };
}

/** A row of for_debug/06_volume/volumes.csv */
export interface VolumeRow {
  name: string;
  is_ref: boolean;
  /** Mesh units³ — exact signed volume when watertight. */
  volume: number;
  method: string;
  /** Oriented bounding box extents, mesh units. obb_a is the VERTICAL axis
   *  (not the largest) — Stage 6 orders them by orientation, not magnitude. */
  obb_a: number;
  obb_b: number;
  obb_c: number;
  /** True when these came from main's AXIS-ALIGNED extents rather than an
   *  oriented box. They are not interchangeable: an AABB around a tilted cube
   *  measures its diagonal, so they must not be used to derive scale. */
  aabb: boolean;
  /** Voxel occupancy cross-check. Should sit a few % ABOVE `volume`. */
  voxel: number | null;
  real_vol_cm3: number;
  real_vol_L: number;
  /** Vertical axis — the one the floor truncates. */
  height_cm: number;
  width_cm: number;
  depth_cm: number;
}

export interface SampleDataset {
  id: string;
  label: string;
  /** What the object actually is, for the samples list. */
  subject: string;
  /** Nominal truth if known, else null. Never invent one. */
  nominalMl: number | null;
  frames: number;
  meshes: {
    leg: string;
    box: string;
    scene: string;
    legNoCut: string;
  };
  volumesCsv: string;
  cuttingLine: string;
}

export type Screen =
  | "samples"
  | "upload"
  | "framing"
  | "processing"
  | "review"
  | "result"
  | "how";

export type ThemeName = "clinical" | "instrument" | "paper";

/** One frame's verdict from stage 0 framing — 00_prep/framing.json.
 *
 * A frame is accepted on either of two paths, and rejected only when neither
 * works:
 *
 *   1. Stage 0 places a full-width square window that holds the whole
 *      reference cube and the marker band, and hands VGGT that crop.
 *   2. Stage 0 cannot place such a window — often a pose where the limb
 *      occludes part of the cube, so its bounds cannot be recovered — but
 *      VGGT's own centre crop would keep what is visible of the reference
 *      anyway. The original frame is passed through untouched.
 *
 * The distinction matters and `mode` carries it: a frame that went through
 * uncropped is a real viewpoint, but whatever VGGT's crop removed is gone.
 *
 * Stage 0 returns one of THREE verdicts per frame, and the distinction is what
 * a defect costs rather than how visible it is:
 *
 *   `pass`    — everything found and framed.
 *   `warning` — used, with a caveat. The band is missing or clipped, or the
 *               cube is clipped. A missing band only means the cut is placed by
 *               a person in review, which it is anyway; a clipped cube falls
 *               back to VGGT's own centre crop. Degraded, not unusable.
 *   `reject`  — not used. The cube was not detected at all, nothing was, or the
 *               file could not be decoded. The reference cube sets the scale of
 *               every number, so its absence cannot be recovered from.
 *
 * Only a reject stops the run. This is what the UI needs in order to say WHICH
 * photo to re-take and why. */
export interface FramingFrame {
  /** Position in the submitted order, 1-based — how the user refers to it. */
  index: number;
  source: string;
  /** The real answer. Absent on reports written before Stage 0 grew the third
   *  verdict, in which case fall back to `accepted`. */
  verdict?: "pass" | "warning" | "reject";
  /** True when the pipeline USES the frame — which includes every warning. */
  accepted: boolean;
  /** Empty when nothing was wrong. A frame can carry a reason and still be
   *  used: a missing band costs the cut, not the scale. */
  reasons: string[];
  /** How much the worst reason matters, within the verdict:
   *  `not crucial`  — the marker band was not found; the cut loses this frame,
   *                   the scale does not.
   *  `crucial`      — the cube or the band is clipped by the window.
   *  `very crucial` — nothing was found, or no cube was. These are the rejects.
   *  Absent when the frame was clean. */
  severity?: "not crucial" | "crucial" | "very crucial" | null;
  /** Whether each object was detected at all, as opposed to detected and then
   *  cut by the window. The two need different remedies. */
  cube_seen?: boolean;
  band_seen?: boolean;
  /** Annotated copy of the source frame: crop window, cube bound, band.
   *  Null when the file could not be decoded, so there is nothing to annotate. */
  overlay: string | null;
  /** The frame handed to VGGT — the crop, or a copy of the original when the
   *  frame passed through uncropped. */
  cropped: string | null;
  /** How the frame was framed:
   *  `crop`                     our window held everything
   *  `crop-clipped`             window centred on the cube; the subject's full
   *                             span did not fit, but cube and band did
   *  `unbounded`                too few cube faces to bound it
   *  `…->uncropped` / `original`  passed through for VGGT to crop itself
   *  Absent on reports written before stage 0 recorded it. */
  mode?: string;
}

export interface FramingReport {
  required: number;
  /** Frames the pipeline will use — passes AND warnings. */
  accepted: number;
  submitted: number;
  /** Every frame clean, no warnings at all. Stricter than `usable`. */
  all_passed: boolean;
  /** No rejects and enough frames — this is what gates a run, not
   *  `all_passed`. Absent on older reports. */
  usable?: boolean;
  /** Sources of the frames carrying a warning, and of the rejected ones. */
  warned?: string[];
  rejected?: string[];
  frames: FramingFrame[];
}
