"""Stage 0 — frame the subject before VGGT ever sees it.

VGGT is handed a 518x518 square. On 9:16 phone photos its default centre crop
resizes the width to 518 and cuts the height to match, which discards 43.8% of
every frame — measured on inputs/small_leg, original rows 840..3000 of 3840
survive. That is not a neutral loss: it clipped the reference cube's base in two
of six frames and put the limb's marker band outside the frame in several, which
is why Stage 3 found only one marker plane instead of two.

The obvious fix, keeping the whole frame by padding, was tested and is not
better in the way that matters. At 518 it improved the global fit — floor
planarity 3.14 -> 2.69 mm RMS, and the two horizontal reference edges agreed
three times more closely, 1.10% -> 0.37% — but the subject shrank from 518 to
291 px across and the limb lost 11% of its volume. At 1022 it failed outright:
the full frame re-admitted the chair and the seated body, and Stage 3 selected
the chair as the reference cube.

So the centre crop was doing two jobs, and only one of them was wasteful. It
threw away pixels, but it also isolated the subject. This stage keeps the second
job and drops the first: find the cube and the limb, and cut a square that
contains both with margin, so the subject fills the frame VGGT is given.

The cube is located by its own ArUco markers rather than by a detector — exact,
free, and it doubles as a scale, since the cube is a known size and its apparent
size tracks distance. The limb needs a model; a person box is too coarse,
because it contains the torso and head and its union with the cube demands a
square larger than the frame, so a segmentation mask is clipped to a band
reaching up from the floor that the cube itself defines.

One caveat this stage cannot fix, and it is the reason CENTRE_ON_SUBJECT
defaults to False. VGGT's pose encoding carries no principal point: it emits
cx = cy = image centre on every frame. An off-centre crop presents it with a
camera it cannot represent, and on inputs/small_leg the subject sits 8.5 degrees
off axis on average and 14.8 at worst. Recentring therefore trades a known pixel
loss for an unknown geometric bias. The window is kept concentric with the frame
by default, and merely tightened around the subject.
"""
import glob
import json
import os

import cv2
import numpy as np

from pipeline.config import IMAGE_EXTENSIONS, REFERENCE_MARKER_DICT
from pipeline.core import vlm_detect as vlm

# The reference cube's edge, and the printed marker's black square, in cm. Only
# their ratio is used here, to recover a face's corners from the marker on it.
FACE_CM = 14.0
MARKER_CM = 6.3

# Fallback height of the marker band above the floor, in cube heights, used only
# when the band cannot be found by colour. The cube stands on the floor, so its
# base fixes floor level and its own height fixes the scale. The band is tied at
# one place on the limb, so its height is a physical constant: measured on
# inputs/small_leg it reads 0.97, 1.13, 1.46 and 1.51 cube heights on the frames
# that resolve it, so 1.6 clears all of them without reaching far past the cut.
LIMB_BAND_CUBE_HEIGHTS = 1.6

# Margin added around the subject union before squaring.
PAD_FRAC = 0.05

# Segmentation model. Nano is enough: the mask only has to bound a limb, not
# delineate it, and it runs in well under a second per frame.
SEG_MODEL = "yolo11n-seg.pt"

# Side length of the emitted square, in pixels. VGGT resizes whatever it is
# given to 518 anyway, and for a square input its arithmetic is exact —
# round(518/14)*14 is 518, so the centre-crop branch never fires and nothing is
# discarded. Doing the reduction here instead means the filter is ours (Lanczos
# holds edges better than bicubic over a 4:1 reduction, and marker corners are
# what the scale check reads), the frames load an order of magnitude faster, and
# no rounding is left to surprise anyone. Set 0 to emit the crop at native
# resolution instead.
OUTPUT_SIZE = 518

# Frames the pipeline needs. VGGT reconstructs from a set, so a frame lost to
# bad framing is not a small loss -- it is a viewpoint the geometry no longer
# has. Every submitted frame must pass AND at least this many must remain, so a
# capture cannot be salvaged by simply discarding whatever failed.
MIN_FRAMES = 6

# Emit this stage's crop, or hand VGGT the original frames.
#
# ON. The point of the stage is that VGGT never gets to choose the crop: it
# centre-crops whatever it is given and discards 44% of a 9:16 photo, with no
# regard for where the reference is. Two frames of inputs/small_leg lose part of
# the cube that way. Cropping here means the window is chosen around the cube
# and the band, and the gate has already verified both survive it.
#
# This was briefly defaulted off, because the limb stopped closing when cropping
# was enabled. That was not the crop's fault: the alpha ladder stopped at 90x
# spacing and the cropped cloud closes at 140x, so the search gave up one rung
# short of the answer. The ladder now runs to 200x and the case terminates.
#
# Frames that fail the gate are never cropped -- a window that cannot hold the
# cube and the band would cut one of them. With --continue-on-rejected they are
# passed through whole and VGGT crops them itself, which is the best available
# treatment for a photo that cannot be framed properly.
CROP_ENABLED = True

# Recentre the window on the subject, or keep it concentric with the frame.
#
# On by default, because frame-centred cropping cannot do the job: a square
# centred on the frame that still contains an off-centre subject has to reach
# the subject's furthest corner, so it comes out at essentially the whole frame
# and is the centre crop again. Measured on inputs/small_leg, frame-centred
# framing left 1 of 6 frames usable against 4 subject-centred.
#
# It is not free. VGGT's pose encoding carries no principal point -- it emits
# cx = cy = image centre on every frame -- so an off-centre window presents it
# with a camera it cannot represent, by 8.5 degrees on average here and 14.8 at
# worst. That is the price of keeping the reference whole, and keeping the
# reference whole matters more: it sets the scale for every reported number.
CENTRE_ON_SUBJECT = True


def _reject_reasons(cube_seen, band_seen, cube_ok, band_ok):
    """Why a frame failed, and how much it matters.

    Two questions, separated because a person acts on them differently: was the
    object SEEN at all, and did it SURVIVE the window.

    The severities are not decoration. A missing band costs the cut but not the
    scale, and the cut only needs the band on some frames -- so a frame without
    one is still worth reconstructing, and is not rejected. A missing or clipped
    cube costs the scale of every number the run reports, and does so with no
    visible sign, which is the whole reason this stage refuses rather than warns.

    Returns (notes, severity). An empty note list means nothing was wrong.
    """
    notes, severity = [], None
    if not cube_seen and not band_seen:
        notes.append("marker and cube missing, very crucial")
        severity = "very crucial"
    elif not cube_seen:
        notes.append("cube missing, crucial")
        severity = "crucial"
    elif not band_seen:
        notes.append("marker missing, not crucial")
        severity = "not crucial"

    # Fit is only a meaningful complaint about something we actually found.
    #
    # Which object was clipped is deliberately not reported. The window is not
    # something a person can adjust -- it is the largest square the photo
    # allows, placed by this stage -- so the remedy is the same either way: step
    # back and re-take. Naming the object would be detail nobody can act on.
    # `cube_ok` and `band_ok` are still recorded in the manifest for debugging.
    clipped_objs = [name for name, seen, good in
                    (("cube", cube_seen, cube_ok), ("band", band_seen, band_ok))
                    if seen and not good]
    if clipped_objs:
        notes.append("objects out of window")
        # Severity still tracks the consequence even though the message does
        # not: a clipped cube corrupts the scale of every reported number, a
        # clipped band only costs the cut.
        severity = "very crucial" if "cube" in clipped_objs else "crucial"
    return notes, severity


def _face_corners(quad, face_cm=FACE_CM, marker_cm=MARKER_CM):
    """The full face's corners, from the marker printed on it.

    The marker sits well inside its face, so its corners are not the cube's.
    Scaling them outward in the image would be wrong -- perspective does not
    preserve distance ratios -- but the marker's own four corners define the
    homography between the face plane and the image, and the face's corners have
    known coordinates in that plane. Mapping them through it is exact.
    """
    hm, hf = marker_cm / 2.0, face_cm / 2.0
    src = np.array([[-hm, -hm], [hm, -hm], [hm, hm], [-hm, hm]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, quad.reshape(4, 2).astype(np.float32))
    dst = np.array([[[-hf, -hf], [hf, -hf], [hf, hf], [-hf, hf]]], dtype=np.float32)
    return cv2.perspectiveTransform(dst, H).reshape(4, 2)


def _cube_faces(gray, dict_name=REFERENCE_MARKER_DICT):
    """Every visible face of the reference, as image-space quads."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return []
    return [_face_corners(q) for q in corners]


def _cube_bbox(gray, image_pil=None, dict_name=REFERENCE_MARKER_DICT):
    """Bounding box of the cube itself, not of the markers on it.

    Sizing on marker corners is what let the crop clip the reference: the
    markers sit well inside their faces and understate the cube by 120-300 px
    per side. Each marker's own corners define the homography onto its face
    plane, though, so the face's corners map back exactly.

    A face still bounds only itself -- the cube continues behind it -- and on
    inputs/small_leg a single face understates the cube by up to 203 px. Two
    adjacent faces share an edge and between them cover most of the silhouette,
    which is why two used to be required. But requiring them rejects a
    legitimate pose where the limb occludes the rest of the reference, and the
    requirement was never really "two faces", it was "a box that contains the
    cube".

    So take the union with an open-vocabulary detection of the box. That box
    alone is not enough either -- measured against the face union it falls
    inside the true silhouette on every frame, by 22 to 90 px -- but the two
    under-cover in different directions, and their union does not. Where three
    faces are visible the detector adds nothing; where one is, it supplies the
    side the face cannot see.
    """
    faces = _cube_faces(gray, dict_name)
    if not faces:
        return None
    p = np.concatenate(faces)
    box = np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
    if image_pil is not None:
        try:
            det, _score = vlm.detect(image_pil, vlm.BOX_PROMPT)
        except Exception:
            det = None
        if det is not None:
            box = np.array([min(box[0], det[0]), min(box[1], det[1]),
                            max(box[2], det[2]), max(box[3], det[3])])
    return box


def _debug_overlay(img, window, cube, band, ok, mode, notes, max_side=1200,
                   effective=None):
    """Annotated copy of the source frame, for inspection only.

    Kept strictly separate from what the pipeline consumes: VGGT is handed the
    clean crop, and these drawn-on frames exist so the framing decision can be
    checked by eye -- particularly for rejected frames, where the point is to
    show which shot to re-take and why.

    `window` is the square this stage proposed; `effective` is the one the
    verdict was actually reached against, which is VGGT's own centre crop
    whenever the proposal could not be used. Drawing only the proposal made
    rejected frames look self-contradictory -- a frame could be marked "cube
    not contained" while the drawn box plainly contained the cube, because the
    box that cut it was VGGT's and was never shown.
    """
    out = img.copy()

    def rect(box, colour, thickness):
        if box is None:
            return
        p0 = (int(round(box[0])), int(round(box[1])))
        p1 = (int(round(box[2])), int(round(box[3])))
        cv2.rectangle(out, p0, p1, colour, thickness)

    differs = (effective is not None
               and not np.allclose(np.asarray(effective, dtype=float),
                                   np.asarray(window, dtype=float), atol=1.0))
    if differs:
        rect(window, (140, 190, 190), 4)     # proposed, not used
        rect(effective, (0, 255, 255), 10)   # what the verdict was reached on
    else:
        rect(window, (0, 255, 255), 10)
    rect(cube, (255, 0, 255), 8)         # reference cube
    rect(band, (0, 255, 0), 8)           # marker band

    scale = max_side / float(max(out.shape[:2]))
    if scale < 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    out = cv2.copyMakeBorder(out, 74, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(out, ("ACCEPTED" if ok else "REJECTED") + f"   [{mode}]",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if ok else (0, 0, 255), 2)
    cv2.putText(out, "; ".join(notes) if notes else
                "cube ok, band ok, both with margin",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(out,
                ("yellow=window VGGT will crop to   grey=our window (unused)   "
                 "magenta=cube   green=band") if differs else
                "yellow=crop window   magenta=cube   green=band",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    return out


def _vggt_window(shape, target=518, patch=14):
    """The region VGGT will keep from a frame handed to it uncropped.

    Declining to crop does not mean nothing is cropped. VGGT resizes the width
    to its input size and centre-crops the height, which on a 9:16 photo throws
    away 44% of the frame -- so the reference can still be cut, just by VGGT
    rather than here. The gate has to test against this window whenever this
    stage is not supplying one of its own.
    """
    h, w = shape[:2]
    new_h = round(h * (target / w) / patch) * patch
    if new_h <= target:
        return np.array([0.0, 0.0, float(w), float(h)])
    start = (new_h - target) // 2
    scale = new_h / h
    return np.array([0.0, start / scale, float(w), (start + target) / scale])


def _grow(box, frac):
    """Expand a box by a fraction of its own larger side."""
    m = frac * max(box[2] - box[0], box[3] - box[1])
    return np.array([box[0] - m, box[1] - m, box[2] + m, box[3] + m], float)


def _leg_mask(image_pil, cube):
    """Mask of the limb being measured.

    A class-based segmenter only knows `person`, so the limb arrives inside a
    whole-body mask and has to be dug out with a nearest-to-the-cube heuristic.
    Asking for "a human leg" by name and segmenting that box returns the limb
    directly. Where several limbs are present -- a second person, the subject's
    own other leg -- the one standing at the reference is the one being
    measured, so proximity to the cube still decides between candidates.
    """
    box, score = vlm.detect(image_pil, vlm.LEG_PROMPT)
    if box is None:
        return None, None
    mask = vlm.segment(image_pil, box)
    if mask is None or not mask.any():
        return None, box
    return mask, box


def _band_bbox(image_pil, bgr, leg_box):
    """The marker band, located by description rather than by colour.

    The previous rule was 2G - R - B > 10, which encodes one khaki band: its own
    config notes that raising the threshold lost the only marker in the dataset,
    and it fired on a houseplant instead of the band on two frames. Naming the
    band works whatever colour it is -- measured on inputs/small_leg it is found
    on 6 of 6 frames at 0.81-0.84, including both frames the colour rule missed.

    Restricted to the limb's own box, so a band-like object elsewhere in the
    scene cannot win.
    """
    box, score = vlm.detect(image_pil, vlm.BAND_PROMPT)
    if box is None:
        return None
    if leg_box is not None:
        # Must overlap the limb it is supposed to be on.
        ix0, iy0 = max(box[0], leg_box[0]), max(box[1], leg_box[1])
        ix1, iy1 = min(box[2], leg_box[2]), min(box[3], leg_box[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
        if inter / area < 0.5:
            return None
    return np.array(box, dtype=float)


def _square_window(union, shape, cube, pad=PAD_FRAC,
                   centre_on_subject=CENTRE_ON_SUBJECT):
    """VGGT's own crop, but free to slide vertically.

    VGGT takes the full width and a centred square of the height. That keeps
    every horizontal pixel and throws away 44% of a 9:16 photo vertically,
    without regard for where the subject is -- which is how the reference cube
    ends up cut on frames where it sits low.

    So keep the shape and drop only the centring: full width, square, and slide
    it up or down until the cube and the band both fit. Nothing is lost
    horizontally at all, and vertically the loss is unavoidable -- the square is
    as tall as the frame is wide, which is the largest square that exists.

    Sliding vertically also costs less than a freely-placed window. VGGT's pose
    encoding carries no principal point and assumes the image centre; a
    full-width window keeps the horizontal principal point exactly right and
    misstates only the vertical, where a subject-centred square got both wrong.

    Returns (window, clipped). `clipped` is True when the subject cannot be made
    to fit at any vertical offset, which is a property of the photograph and
    means the frame has to be re-taken.
    """
    h, w = shape
    side = float(min(w, h))
    if side >= h:                      # already square or wider than tall
        return np.array([0.0, 0.0, float(w), float(h)]), False

    # What has to survive, with its margin already applied by the caller.
    top, bottom = float(union[1]), float(union[3])
    needed = bottom - top
    clipped = needed > side

    if clipped:
        # Cannot hold it all. Favour the cube: it sets the scale, and the limb
        # above the band is discarded by the cut anyway.
        centre = (float(cube[1]) + float(cube[3])) / 2.0
    else:
        centre = (top + bottom) / 2.0

    y0 = float(np.clip(centre - side / 2.0, 0.0, h - side))
    return np.array([0.0, y0, float(w), y0 + side]), clipped


def prepare_frames(image_folder, out_dir, band_heights=LIMB_BAND_CUBE_HEIGHTS,
                   pad=PAD_FRAC, centre_on_subject=CENTRE_ON_SUBJECT,
                   seg_model=SEG_MODEL, output_size=OUTPUT_SIZE, strict=True,
                   min_frames=MIN_FRAMES, crop=CROP_ENABLED):
    """Crop every frame to its subject; return the manifest.

    A frame that cannot be cropped to hold the whole reference and the marker
    band is rejected rather than trimmed to fit. Silently handing VGGT a
    truncated reference is the worst outcome available: the cube sets the scale
    for every number the pipeline reports, and a frame that clips it degrades
    the result invisibly. Re-shooting that one camera pose from further back is
    cheap; an unexplained few percent of volume error is not.
    """
    from PIL import Image

    paths = sorted(p for p in glob.glob(os.path.join(image_folder, "*"))
                   if p.lower().endswith(IMAGE_EXTENSIONS))
    if not paths:
        raise SystemExit(f"ERROR: no images in {image_folder}")

    os.makedirs(out_dir, exist_ok=True)
    debug_dir = os.path.join(out_dir, "..", "for_debug")
    os.makedirs(debug_dir, exist_ok=True)
    records = []
    band_colours = []
    print("=" * 60)
    print(f"STAGE 0: framing {len(paths)} frames  "
          f"(band={band_heights} cube heights, pad={pad:.0%}, "
          f"{'subject-centred' if centre_on_subject else 'frame-centred'}, "
          f"out={output_size or 'native'})")
    print("=" * 60)

    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            print(f"  {os.path.basename(path)}: unreadable — skipped")
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        faces = _cube_faces(gray)
        cube = _cube_bbox(gray, image_pil)

        # Find the limb and the band regardless of whether the cube can be
        # bounded. These are independent measurements, and skipping them when
        # the cube is short of two faces reported "band not found" on a frame
        # whose band detects perfectly well at 0.74 confidence -- the band was
        # never looked for.
        leg, leg_box = _leg_mask(image_pil, cube)
        band = _band_bbox(image_pil, img, leg_box)
        if band is not None:
            colour = vlm.trace_band_colour(img, band)
            if colour:
                band_colours.append(colour)

        if cube is not None:
            cube_h = cube[3] - cube[1]
            band_top = (band[1] if band is not None
                        else cube[3] - band_heights * cube_h)
            union = _grow(cube, pad)
            if band is not None:
                b = _grow(band, pad)
                union = np.array([min(union[0], b[0]), min(union[1], b[1]),
                                  max(union[2], b[2]), max(union[3], b[3])])
            if leg is not None:
                strip = leg[max(0, int(band_top)):int(cube[3]), :]
                if strip.any():
                    xs = np.where(strip.any(axis=0))[0]
                    union = np.array([min(union[0], xs.min()), min(union[1], band_top),
                                      max(union[2], xs.max()), max(union[3], cube[3])])
            union[1] = min(union[1], band_top)
            window, clipped = _square_window(union, (h, w), cube, pad,
                                             centre_on_subject)
            mode = "crop-clipped" if clipped else "crop"
        else:
            # Too few faces to bound the cube. That only matters if this stage
            # is going to cut a window; if it is not, nothing can be clipped by
            # us and the frame is simply a viewpoint where the limb happens to
            # occlude part of the reference.
            side = float(min(h, w))
            window = np.array([(w - side) / 2, (h - side) / 2,
                               (w + side) / 2, (h + side) / 2], dtype=float)
            clipped = False
            mode = "unbounded"

        # Two different questions, previously run together: can this frame be
        # cropped, and is it usable at all. A pose where the limb occludes part
        # of the reference cannot be cropped -- one visible face does not bound
        # a cube -- but it is still a real viewpoint, and VGGT's own crop may
        # keep everything that matters. So: prefer our window, fall back to
        # VGGT's, and reject only when the reference survives neither.
        def _fits_window(win, box, margin_frac=pad):
            if box is None:
                return False
            m = margin_frac * max(box[2] - box[0], box[3] - box[1])
            tol = 1.0
            return bool(box[0] >= win[0] + m - tol and box[1] >= win[1] + m - tol
                        and box[2] <= win[2] - m + tol
                        and box[3] <= win[3] - m + tol)

        seen = np.concatenate(faces) if faces else None
        visible_cube = (None if seen is None else
                        np.array([seen[:, 0].min(), seen[:, 1].min(),
                                  seen[:, 0].max(), seen[:, 1].max()]))

        can_crop = (crop and cube is not None and len(faces) >= 1
                    and _fits_window(window, cube)
                    and band is not None and _fits_window(window, band))

        vggt_win = _vggt_window(img.shape)
        # Uncropped is acceptable when whatever IS visible of the reference
        # survives VGGT's crop, and any band we found does too.
        can_pass_through = (_fits_window(vggt_win, visible_cube)
                            and (band is None or _fits_window(vggt_win, band)))

        ok = bool(can_crop or can_pass_through)
        cube_ok = bool(can_crop or _fits_window(vggt_win, visible_cube))
        band_ok = bool(band is None or _fits_window(window if can_crop else vggt_win,
                                                    band))
        if can_crop:
            mode = "crop-clipped" if clipped else "crop"
        elif can_pass_through:
            mode = f"{mode}->uncropped" if crop else "original"
        x0, y0, x1, y1 = [int(round(v)) for v in
                          (window if can_crop else vggt_win)]

        wx0, wy0, wx1, wy1 = [int(round(v)) for v in window]
        out_path = os.path.join(out_dir, f"frame_{i:02d}.png")
        if can_crop:
            cropped = img[wy0:wy1, wx0:wx1]
            if output_size and cropped.shape[0] != output_size:
                cropped = cv2.resize(cropped, (output_size, output_size),
                                     interpolation=cv2.INTER_AREA if
                                     cropped.shape[0] > output_size
                                     else cv2.INTER_LANCZOS4)
            cv2.imwrite(out_path, cropped)
        else:
            # Anything we cannot frame ourselves goes to VGGT untouched.
            #
            # That covers both the usable case -- a pose where the cube's bounds
            # cannot be recovered but VGGT's own centre crop keeps what matters
            # -- and every rejection. A rejected frame is still written so it can
            # be inspected, and so that continuing past the gate does not need a
            # second pass over the inputs. Cropping a frame we have already
            # judged unframeable would only choose, arbitrarily, which part of
            # the evidence to destroy.
            cv2.imwrite(out_path, img)
            if not can_pass_through:
                mode = f"{mode}->uncropped" if crop else "original"

        cube_seen = cube is not None
        band_seen = band is not None
        notes, severity = _reject_reasons(cube_seen, band_seen, cube_ok, band_ok)
        cv2.imwrite(
            os.path.join(debug_dir,
                         f"{i:02d}_{os.path.splitext(os.path.basename(path))[0]}.png"),
            _debug_overlay(img, window, cube, band, ok, mode, notes,
                           effective=(window if can_crop else vggt_win)))

        off_px = np.hypot((x0 + x1) / 2 - w / 2, (y0 + y1) / 2 - h / 2)

        records.append({
            "index": i + 1, "source": os.path.basename(path),
            "overlay": f"{i:02d}_{os.path.splitext(os.path.basename(path))[0]}.png",
            "reasons": notes, "output": os.path.basename(out_path),
            "frame_size": [w, h], "window": [x0, y0, x1, y1],
            "cube_bbox": None if cube is None else cube.tolist(),
            "band_bbox": None if band is None else band.tolist(),
            "clipped": bool(clipped),
            "offset_px": float(off_px), "mode": mode,
            "cube_ok": cube_ok, "band_ok": band_ok,
            "cube_seen": cube_seen, "band_seen": band_seen,
            "severity": severity,
        })
        verdict = "ok" if (cube_ok and band_ok) else "REJECTED"
        print(f"  {os.path.basename(path):<16} -> {os.path.basename(out_path)}  "
              f"{mode:<13} {x1-x0}px ({(x1-x0)/min(w,h):.0%} of frame)  {verdict}"
              f"{('  [' + severity + ']') if severity else ''}"
              f"{('  ' + '; '.join(notes)) if notes else ''}")

    marker_colour = None
    if band_colours:
        arr = np.array([c["bgr"] for c in band_colours], dtype=float)
        med = np.median(arr, axis=0)
        b, g, r = (float(v) for v in med)
        limb = np.median(np.array([c["limb_bgr"] for c in band_colours
                                   if "limb_bgr" in c], dtype=float), axis=0) \
            if any("limb_bgr" in c for c in band_colours) else None
        marker_colour = {
            "limb_rgb": None if limb is None else
                        [round(float(limb[2]), 1), round(float(limb[1]), 1),
                         round(float(limb[0]), 1)],
            "bgr": [round(b, 1), round(g, 1), round(r, 1)],
            "rgb": [round(r, 1), round(g, 1), round(b, 1)],
            "exg": round(2 * g - r - b, 1),
            "hsv": band_colours[len(band_colours) // 2]["hsv"],
            "n_frames": len(band_colours),
        }
        print(f"\n  Marker colour learned from {len(band_colours)} frames: "
              f"RGB {[int(v) for v in marker_colour['rgb']]}, "
              f"excess-green {marker_colour['exg']:+.0f} "
              f"— handed to Stage 3 instead of a fixed threshold")

    bad = [r for r in records if not (r["cube_ok"] and r["band_ok"])]
    good = [r for r in records if r["cube_ok"] and r["band_ok"]]

    manifest = {
        "source": os.path.abspath(image_folder),
        "rejected": [r["source"] for r in bad],
        "marker_colour": marker_colour,
        "band_cube_heights": band_heights,
        "output_size": output_size,
        "pad_frac": pad,
        "centre_on_subject": centre_on_subject,
        "frames": records,
    }
    with open(os.path.join(out_dir, "..", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # A second, narrower file for anything presenting this to a person. The
    # manifest carries pixel geometry that only the pipeline cares about; this
    # carries the verdict, the reason and the picture that shows it, which is
    # what a reviewer needs in order to know which shot to re-take.
    report = {
        "required": min_frames,
        "accepted": len(good),
        "submitted": len(records),
        "all_passed": not bad and len(good) >= min_frames,
        "marker_colour": marker_colour,
        "frames": [{
            "index": r["index"],
            "source": r["source"],
            "accepted": bool(r["cube_ok"] and r["band_ok"]),
            "reasons": r["reasons"],
            "severity": r["severity"],
            "cube_seen": r["cube_seen"],
            "band_seen": r["band_seen"],
            "overlay": r["overlay"],
            "cropped": r["output"],
            # HOW the frame passed, not just whether. A frame can be accepted
            # on either of two paths -- our window, or VGGT's own centre crop
            # when ours cannot be placed -- and those are not equally good. A
            # reviewer told only "accepted" cannot tell that a frame went
            # through uncropped and lost whatever VGGT's crop removed.
            "mode": r["mode"],
        } for r in records],
    }
    with open(os.path.join(out_dir, "..", "framing.json"), "w") as f:
        json.dump(report, f, indent=2)


    manifest["accepted"] = len(good)
    manifest["rejected_detail"] = [
        {"index": r["index"], "source": r["source"],
         "cube_ok": r["cube_ok"], "band_ok": r["band_ok"]} for r in bad]

    if not band_colours:
        print("\n  WARNING: the marker band was not found on any frame. Stage 3 "
              "will fall back to\n  the colour defaults in pipeline/config.py, "
              "which describe one particular khaki\n  band — if your marker is "
              "another colour the cut will not find it.")

    if strict and (bad or len(good) < min_frames):
        print()
        print("=" * 60)
        print("STAGE 0: INPUT NOT ACCEPTED — please re-take and re-submit")
        print("=" * 60)
        print(f"  {len(good)} of {len(records)} frames usable; "
              f"{min_frames} are required, and every submitted frame must pass.")
        if bad:
            print(f"  Re-take: "
                  + ", ".join(f"img{r['index']} ({r['source']})" for r in bad))
        print()
        # Grouped by severity, because the remedies are different. These read
        # off the same reasons the report carries, rather than re-deriving them
        # from bounding boxes -- which is how the old version came to describe a
        # cube that was never detected as one that "wasn't contained".
        by_sev = {}
        for r in bad:
            by_sev.setdefault(r["severity"], []).append(r)

        def _listing(rows):
            for r in rows:
                print(f"    img{r['index']}  ({r['source']})")
            print()

        clipped = [r for r in bad if any("out of window" in n for n in r["reasons"])]
        if clipped:
            print("  Shot too close — the whole reference cube and the marker "
                  "band, each with")
            print(f"  {pad:.0%} margin, must fit one square. The largest square "
                  f"available is the")
            print("  photo's short side, so this cannot be cropped to, only "
                  "re-framed.")
            print("  Re-take from further back:")
            _listing(clipped)

        no_cube = [r for r in bad if not r["cube_seen"]]
        if no_cube:
            print("  The reference cube was not found at all. It sets the scale "
                  "for every number")
            print("  the run reports, so a frame without it cannot be used. "
                  "Re-take with the cube")
            print("  in shot, showing a corner rather than one face straight on:")
            _listing(no_cube)

        if not bad and len(good) < min_frames:
            print(f"  All frames passed, but only {len(good)} were supplied. "
                  f"Add {min_frames - len(good)} more")
            print("  view(s) around the subject and re-submit.")
            print()
        print("  The pipeline will not run until every frame passes and at "
              f"least {min_frames} are")
        print("  supplied. Re-submit the corrected images and run stage 0 "
              "again.")
        print("  (--continue-on-rejected runs anyway, for experiments only.)")
        raise SystemExit(1)

    vlm.release()
    return manifest
