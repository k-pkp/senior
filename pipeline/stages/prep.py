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
import math
import os

import cv2
import numpy as np

from pipeline.config import IMAGE_EXTENSIONS, REFERENCE_MARKER_DICT
from pipeline.core import vlm_detect as vlm

# The reference cube's edge, and the printed marker's black square, in cm. Only
# their ratio is used here, to recover a face's corners from the marker on it.
FACE_CM = 14.0
MARKER_CM = 6.3

# Largest a detected marker band may be, as a fraction of the limb's mask area.
# An open-vocabulary detector always returns its best candidate for "cord", and
# on a capture with no cord its best candidate is the leg. Measured band/limb
# area ratios: real bands on inputs/small_leg 0.04-0.07; false positives on
# inputs/est_325 1.23 and inputs/short_leg 2.19-2.91. 0.35 is five times the
# largest real band and a third of the smallest false one. See _band_bbox.
BAND_MAX_LIMB_FRAC = 0.35

# Pad around the limb box before running the band detector on a crop of it. The
# band sits at the leg's edge, so a tight box would cut it off, and the pad keeps
# the leg context the prompt ("on the leg") relies on. As a fraction of the box's
# larger side. See _band_bbox.
BAND_CROP_PAD_FRAC = 0.2

# What fraction of the submitted frames must independently show the band before
# its colour is trusted, rounded up.
#
# A fixed count does not scale: two detections out of six is corroboration, two
# out of twenty is noise. A fraction asks the right question -- does most of the
# capture agree that there is a band?
#
# What it is guarding against, measured on inputs/short_leg, a capture with no
# marker at all: a single 74x60 px false positive on 1 of 8 frames taught the
# pipeline that the marker is RGB(217,207,198), which is the floor tile. Stage 3
# then found 198 "band" points and cut the limb at 61% of its height. With no
# colour at all Stage 3 correctly finds no plane.
#
# At 0.6 the bar is 4 of 6 frames, or 5 of 8. The real band on inputs/small_leg
# is found on 6 of 6; the false positive on short_leg on 1 of 8.
BAND_MIN_FRAME_FRAC = 0.6

# Fallback height of the marker band above the floor, in cube heights, used only
# when the band cannot be found by colour. The cube stands on the floor, so its
# base fixes floor level and its own height fixes the scale. The band is tied at
# one place on the limb, so its height is a physical constant: measured on
# inputs/small_leg it reads 0.97, 1.13, 1.46 and 1.51 cube heights on the frames
# that resolve it, so 1.6 clears all of them without reaching far past the cut.
LIMB_BAND_CUBE_HEIGHTS = 1.6

# Margin added around the subject union before squaring.
PAD_FRAC = 0.05

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


# Stage 0's three possible verdicts on a frame. They are ordered by severity so
# the worst finding on a frame wins, and so a run's overall state is just the
# maximum over its frames.
PASS, WARNING, REJECT = "pass", "warning", "reject"
_VERDICT_RANK = {PASS: 0, WARNING: 1, REJECT: 2}


def _frame_verdict(cube_seen, band_seen, cube_ok, band_ok):
    """What to do about a frame: use it, use it with a caveat, or refuse it.

    The distinction that matters is **what a defect costs**, not how visible it
    is:

      - **Nothing detected, or no reference cube detected.** There is nothing
        to measure against and no way to recover downstream. REJECT — the run
        does not start.
      - **Detected but clipped by the window.** The object is there and the
        frame is real; what is lost is crop quality, because VGGT falls back to
        its own centre crop. Degraded, not unusable. WARNING.
      - The **band only decides where the limb is cut.** A capture with no band
        is a perfectly good capture that will not be cut automatically — the
        person places the plane in the review step instead. That is a WARNING:
        the frame goes through, and the run is told why it might not be able to
        cut.

    A clipped band is likewise a WARNING, not a rejection: the cut it informs is
    confirmed by a person before it is applied, so a bad band costs a worse
    starting suggestion, not a wrong answer.

    Returns (verdict, notes, severity).
    """
    notes, severity, verdict = [], None, PASS

    def escalate(level, note, sev):
        nonlocal verdict, severity
        notes.append(note)
        if _VERDICT_RANK[level] > _VERDICT_RANK[verdict]:
            verdict = level
        severity = sev

    if not cube_seen and not band_seen:
        escalate(REJECT, "nothing detected — no cube and no marker", "very crucial")
    elif not cube_seen:
        escalate(REJECT, "cube missing — the scale cannot be recovered", "very crucial")
    elif not band_seen:
        escalate(WARNING, "marker missing — the cut must be placed by hand", "not crucial")

    # Fit is only a meaningful complaint about something actually found.
    if cube_seen and not cube_ok:
        # A WARNING and not a REJECT: the cube was found, so the frame is a real
        # viewpoint. What fails is only this stage's crop, and VGGT's own centre
        # crop takes over. That is worth saying loudly — a clipped reference
        # costs scale — but it is not grounds for refusing a frame that exists.
        escalate(WARNING, "cube out of window — VGGT will centre-crop instead; "
                          "re-take from further back if you can", "crucial")
    if band_seen and not band_ok:
        escalate(WARNING, "marker out of window — the suggested cut may be off", "crucial")

    return verdict, notes, severity


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


def _mask_bbox(mask):
    """Tight box around a boolean mask, or None if it is empty."""
    if mask is None or not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return np.array([cols[0], rows[0], cols[-1], rows[-1]], dtype=float)


def _debug_overlay(img, window, cube, band, ok, mode, notes, max_side=1200,
                   effective=None, limb=None, verdict=None):
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
    # Four boxes, so a person can see every decision the stage made:
    #   magenta  the reference cube      yellow  the window that was used
    #   orange   the limb being measured green   the marker band
    # The limb is drawn because without it a missing band is unreadable —
    # you cannot tell whether the detector found the wrong thing or nothing.
    rect(limb, (0, 140, 255), 6)         # limb
    rect(cube, (255, 0, 255), 8)         # reference cube
    rect(band, (0, 255, 0), 8)           # marker band

    scale = max_side / float(max(out.shape[:2]))
    if scale < 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    out = cv2.copyMakeBorder(out, 96, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    banner = {"pass": ("PASS", (0, 255, 0)),
              "warning": ("WARNING", (0, 200, 255)),
              "reject": ("REJECT", (0, 0, 255))}.get(
                  verdict, ("ACCEPTED", (0, 255, 0)) if ok else ("REJECTED", (0, 0, 255)))
    cv2.putText(out, f"{banner[0]}   [{mode}]",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner[1], 2)

    cv2.putText(out, "; ".join(notes) if notes else
                "cube ok, band ok, both with margin",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(out,
                ("yellow=window VGGT will crop to   grey=our window (unused)   "
                 "magenta=cube   orange=limb   green=band") if differs else
                "yellow=crop window   magenta=cube   orange=limb   green=band",
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


def _band_bbox(image_pil, bgr, leg_box, limb_mask=None):
    """The marker band, located by description rather than by colour.

    The previous rule was 2G - R - B > 10, which encodes one khaki band: its own
    config notes that raising the threshold lost the only marker in the dataset,
    and it fired on a houseplant instead of the band on two frames. Naming the
    band works whatever colour it is -- measured on inputs/small_leg it is found
    on 6 of 6 frames at 0.81-0.84, including both frames the colour rule missed.

    Two guards, and the second was added after it went wrong:

    1. The box must overlap the limb, so a band-like object elsewhere in the
       scene cannot win.
    2. **The box must be small relative to the limb.** An open-vocabulary
       detector always returns its best candidate for "cord", and on a capture
       with no cord at all its best candidate is the leg itself -- which passes
       guard 1 trivially, because the leg is entirely on the leg.

    Guard 2 is measured, not guessed. Band area as a fraction of the limb's mask
    area, over three datasets:

        inputs/small_leg (a real band)   0.04 - 0.07
        inputs/est_325   (no band)       1.23
        inputs/short_leg (no band)       2.19 - 2.91

    A cord tied around a limb cannot be larger than the limb. BAND_MAX_LIMB_FRAC
    sits at 0.35 -- five times the largest real band seen, and a third of the
    smallest false positive.

    What this cost before it was fixed: on inputs/short_leg the hallucinated band
    inflated the crop window until nothing fit, and Stage 0 rejected 5 of 8
    frames as `objects out of window`. With the guard, all 8 pass. It also fed
    `trace_band_colour` a box that is mostly skin, so the "band" colour handed to
    Stage 3 was the limb's own colour -- leaving the discriminant with no
    contrast to separate them by.
    """
    box, score = vlm.detect(image_pil, vlm.BAND_PROMPT)
    if leg_box is not None:
        # Re-run on a padded crop of the limb. The full-image pass is the
        # fallback: it keeps distractors (a houseplant, another limb) in view,
        # and cropping removes them. The crop is padded, not tight, so the band
        # at the leg's edge survives and the "on the leg" context is retained.
        crop_box = _grow(leg_box, BAND_CROP_PAD_FRAC)
        l, t = int(max(0, crop_box[0])), int(max(0, crop_box[1]))
        r, b = int(min(image_pil.size[0], crop_box[2])), int(min(image_pil.size[1], crop_box[3]))
        if r - l > 16 and b - t > 16:
            cbox, cscore = vlm.detect(image_pil.crop((l, t, r, b)), vlm.BAND_PROMPT)
            if cbox is not None:
                box = [cbox[0] + l, cbox[1] + t, cbox[2] + l, cbox[3] + t]
                score = cscore
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
    if limb_mask is not None and limb_mask.any():
        band_area = (box[2] - box[0]) * (box[3] - box[1])
        ratio = band_area / float(limb_mask.sum())
        if ratio > BAND_MAX_LIMB_FRAC:
            print(f"  band rejected: {ratio:.2f}x the limb's area "
                  f"(max {BAND_MAX_LIMB_FRAC}) — the detector returned the limb, "
                  f"not a cord on it")
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


def _box_fits_inside(window, box, margin_frac):
    """Does `box` sit inside `window` with a margin, in pixels?

    The margin scales with the box's own larger side rather than being a fixed
    number of pixels, so the same rule means the same thing on a cube filling
    the frame and on one far away. A box of None never fits: "not detected" and
    "detected but clipped" are different answers and only one of them is a fit.
    """
    if box is None:
        return False
    margin = margin_frac * max(box[2] - box[0], box[3] - box[1])
    # One pixel of slack, so a box touching the window edge exactly is not
    # rejected by floating-point rounding alone.
    tolerance = 1.0
    return bool(box[0] >= window[0] + margin - tolerance
                and box[1] >= window[1] + margin - tolerance
                and box[2] <= window[2] - margin + tolerance
                and box[3] <= window[3] - margin + tolerance)


def _average_band_colour(per_frame_colours, total_frames=None, min_frac=None):
    """One marker colour for the capture, from the per-frame traces.

    Median rather than mean across frames: a frame where the trace wandered off
    the cord reports a wildly different colour, and one such frame would drag a
    mean far enough to matter. Stage 3 receives this instead of the fixed khaki
    thresholds in config, which is what lets a band of any colour work.

    Unless most of the capture agrees there is a band -- BAND_MIN_FRAME_FRAC of
    the submitted frames, rounded up -- the colour is discarded and None is
    returned, so Stage 3 falls back to finding no band at all. For a capture
    with no band that is the right answer, and it is safe besides: the cut is
    placed by a person in the review step either way.
    """
    if not per_frame_colours:
        return None
    if total_frames:
        frac = BAND_MIN_FRAME_FRAC if min_frac is None else min_frac
        needed = int(math.ceil(frac * total_frames))
        if len(per_frame_colours) < needed:
            print(f"  marker colour DISCARDED: the band was found on "
                  f"{len(per_frame_colours)} of {total_frames} frames, and "
                  f"{needed} are needed ({frac:.0%}) to corroborate it. A "
                  f"minority detection is not evidence — Stage 3 will look for "
                  f"no band, and the cut is placed by hand.")
            return None

    median_bgr = np.median(
        np.array([c["bgr"] for c in per_frame_colours], dtype=float), axis=0)
    blue, green, red = (float(v) for v in median_bgr)

    with_limb = [c["limb_bgr"] for c in per_frame_colours if "limb_bgr" in c]
    limb_bgr = (np.median(np.array(with_limb, dtype=float), axis=0)
                if with_limb else None)

    return {
        # The limb's own colour matters as much as the band's: what separates
        # them is the contrast between the two, not either one alone.
        "limb_rgb": None if limb_bgr is None else
                    [round(float(limb_bgr[2]), 1), round(float(limb_bgr[1]), 1),
                     round(float(limb_bgr[0]), 1)],
        "bgr": [round(blue, 1), round(green, 1), round(red, 1)],
        "rgb": [round(red, 1), round(green, 1), round(blue, 1)],
        "exg": round(2 * green - red - blue, 1),
        "hsv": per_frame_colours[len(per_frame_colours) // 2]["hsv"],
        "n_frames": len(per_frame_colours),
    }


def _write_frame_for_vggt(image, window, should_crop, output_size, out_path):
    """Write the frame VGGT will read, cropped or whole.

    Anything not croppable goes out untouched -- both the usable case, where
    VGGT's own centre crop keeps what matters, and every rejection. Cropping a
    frame already judged unframeable would only choose, arbitrarily, which part
    of the evidence to destroy.
    """
    if not should_crop:
        cv2.imwrite(out_path, image)
        return

    left, top, right, bottom = [int(round(v)) for v in window]
    cropped = image[top:bottom, left:right]
    if output_size and cropped.shape[0] != output_size:
        # Reducing: INTER_AREA averages the discarded pixels instead of dropping
        # them. Enlarging: Lanczos holds marker edges, which the scale check reads.
        shrinking = cropped.shape[0] > output_size
        cropped = cv2.resize(cropped, (output_size, output_size),
                             interpolation=cv2.INTER_AREA if shrinking
                             else cv2.INTER_LANCZOS4)
    cv2.imwrite(out_path, cropped)


def _read_image_bgr(path):
    """Decode a photograph to BGR, HEIC included.

    `cv2.imread` cannot open HEIC at all, and a phone shooting in its default
    format produces nothing else. Stage 1 registers the HEIC opener inside
    `vggt/utils/load_fn.py`, but Stage 0 never imported it — so every frame of a
    HEIC capture arrived here as None and was recorded `file unreadable`. On
    inputs/est_325 that was all 8 of 8 frames: the framing gate had never
    actually looked at that dataset.

    Returns None only when the file genuinely cannot be decoded.
    """
    img = cv2.imread(path)
    if img is not None:
        return img

    from PIL import Image as _Image
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        print(f"  {os.path.basename(path)}: OpenCV cannot decode it and "
              f"pillow-heif is not installed")
        return None
    try:
        with _Image.open(path) as handle:
            return cv2.cvtColor(np.array(handle.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        # Say why. A bare `return None` here is how a whole HEIC dataset came to
        # be reported as unreadable without anyone learning the reason.
        print(f"  {os.path.basename(path)}: {type(exc).__name__}: {exc}")
        return None


def _locate_subject(image_bgr, image_pil):
    """Everything this stage needs to find in one photograph.

    The band is looked for whether or not the cube could be bounded. Skipping it
    when the cube was short of faces once reported "band not found" on a frame
    whose band detects perfectly well at 0.74 confidence -- it had simply never
    been looked for. They are independent measurements and are taken as such.

    Returns (cube face quads, cube box, limb mask, band box, band colour).
    """
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_quads = _cube_faces(grey)
    cube_box = _cube_bbox(grey, image_pil)

    limb_mask, limb_box = _leg_mask(image_pil, cube_box)
    band_box = _band_bbox(image_pil, image_bgr, limb_box, limb_mask)

    band_colour = None
    if band_box is not None:
        band_colour = vlm.trace_band_colour(image_bgr, band_box)
    return face_quads, cube_box, limb_mask, band_box, band_colour


def _subject_bounds(cube_box, band_box, limb_mask, band_heights, pad):
    """The region a crop has to keep: the cube, the band, and the limb between.

    The limb is included only over the span from the band down to the cube's
    base. Its full mask reaches the thigh and the torso, and a square containing
    those is larger than the photograph.

    When no band was found its height is estimated from the cube instead: the
    cube stands on the floor so its base fixes floor level and its height fixes
    the scale, and the band is tied at one place on the limb.
    """
    cube_height = cube_box[3] - cube_box[1]
    band_top = (band_box[1] if band_box is not None
                else cube_box[3] - band_heights * cube_height)

    bounds = _grow(cube_box, pad)
    if band_box is not None:
        padded_band = _grow(band_box, pad)
        bounds = np.array([min(bounds[0], padded_band[0]),
                           min(bounds[1], padded_band[1]),
                           max(bounds[2], padded_band[2]),
                           max(bounds[3], padded_band[3])])

    if limb_mask is not None:
        limb_strip = limb_mask[max(0, int(band_top)):int(cube_box[3]), :]
        if limb_strip.any():
            occupied_columns = np.where(limb_strip.any(axis=0))[0]
            bounds = np.array([min(bounds[0], occupied_columns.min()),
                               min(bounds[1], band_top),
                               max(bounds[2], occupied_columns.max()),
                               max(bounds[3], cube_box[3])])
    bounds[1] = min(bounds[1], band_top)
    return bounds


def prepare_frames(image_folder, out_dir, band_heights=LIMB_BAND_CUBE_HEIGHTS,
                   pad=PAD_FRAC, centre_on_subject=CENTRE_ON_SUBJECT,
                   output_size=OUTPUT_SIZE, strict=True,
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

    for frame_index, path in enumerate(paths):
        img = _read_image_bgr(path)
        if img is None:
            # Record it, do not drop it. Skipping silently removed the frame from
            # the report altogether: `submitted` under-counted the photos the user
            # actually gave, the frame numbering gained a gap, and the one thing
            # this stage exists to say -- which photo to re-take -- went unsaid
            # for the photo that most needed re-taking.
            print(f"  {os.path.basename(path):<16} UNREADABLE — cannot be decoded")
            records.append({
                "index": frame_index + 1, "source": os.path.basename(path),
                "overlay": None, "reasons": ["file unreadable — cannot be decoded"],
                "output": None, "frame_size": None, "window": [0, 0, 0, 0],
                "cube_bbox": None, "band_bbox": None, "clipped": False,
                "offset_px": 0.0, "mode": "unreadable",
                "cube_ok": False, "band_ok": False,
                "cube_seen": False, "band_seen": False,
                "severity": "very crucial", "verdict": REJECT,
            })
            continue
        frame_height, frame_width = img.shape[:2]
        image_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        faces, cube, leg, band, colour = _locate_subject(img, image_pil)
        if colour:
            band_colours.append(colour)

        if cube is not None:
            union = _subject_bounds(cube, band, leg, band_heights, pad)
            window, clipped = _square_window(union, (frame_height, frame_width),
                                             cube, pad,
                                             centre_on_subject)
            mode = "crop-clipped" if clipped else "crop"
        else:
            # Too few faces to bound the cube. That only matters if this stage
            # is going to cut a window; if it is not, nothing can be clipped by
            # us and the frame is simply a viewpoint where the limb happens to
            # occlude part of the reference.
            side = float(min(frame_height, frame_width))
            window = np.array([(frame_width - side) / 2, (frame_height - side) / 2,
                               (frame_width + side) / 2, (frame_height + side) / 2],
                              dtype=float)
            clipped = False
            mode = "unbounded"

        # Two different questions, previously run together: can this frame be
        # cropped, and is it usable at all. A pose where the limb occludes part
        # of the reference cannot be cropped -- one visible face does not bound
        # a cube -- but it is still a real viewpoint, and VGGT's own crop may
        # keep everything that matters. So: prefer our window, fall back to
        # VGGT's, and reject only when the reference survives neither.
        def _fits(window, box):
            return _box_fits_inside(window, box, pad)

        visible_face_corners = np.concatenate(faces) if faces else None
        visible_cube = (None if visible_face_corners is None else
                        np.array([visible_face_corners[:, 0].min(), visible_face_corners[:, 1].min(),
                                  visible_face_corners[:, 0].max(), visible_face_corners[:, 1].max()]))

        # The band is OPTIONAL and must not block cropping. It used to read
        # `band is not None and _fits(window, band)`, which made a band a
        # precondition for cropping at all -- so on any capture without a marker
        # (inputs/est_325 has none by design) `can_crop` was always False and the
        # stage silently degraded to VGGT's own centre crop, which is the exact
        # failure Stage 0 exists to prevent. The stage's own rejection table has
        # always said a missing band "costs the cut, not the scale".
        can_crop = (crop and cube is not None and len(faces) >= 1
                    and _fits(window, cube)
                    and (band is None or _fits(window, band)))

        vggt_win = _vggt_window(img.shape)
        # Uncropped is acceptable when whatever IS visible of the reference
        # survives VGGT's crop, and any band we found does too.
        can_pass_through = (_fits(vggt_win, visible_cube)
                            and (band is None or _fits(vggt_win, band)))

        usable = bool(can_crop or can_pass_through)
        cube_ok = bool(can_crop or _fits(vggt_win, visible_cube))
        band_ok = bool(band is None or _fits(window if can_crop else vggt_win,
                                                    band))
        if can_crop:
            mode = "crop-clipped" if clipped else "crop"
        elif can_pass_through:
            mode = f"{mode}->uncropped" if crop else "original"
        left, top, right, bottom = [int(round(edge)) for edge in
                          (window if can_crop else vggt_win)]

        out_path = os.path.join(out_dir, f"frame_{frame_index:02d}.png")
        _write_frame_for_vggt(img, window, can_crop, output_size, out_path)
        if not can_crop and not can_pass_through:
            # Rejected AND not croppable: the frame still goes out whole so it
            # can be inspected, and the mode records that VGGT will be the one
            # doing the cropping.
            mode = f"{mode}->uncropped" if crop else "original"

        cube_seen = cube is not None
        band_seen = band is not None
        verdict, notes, severity = _frame_verdict(cube_seen, band_seen,
                                                  cube_ok, band_ok)
        cv2.imwrite(
            os.path.join(debug_dir,
                         f"{frame_index:02d}_{os.path.splitext(os.path.basename(path))[0]}.png"),
            _debug_overlay(img, window, cube, band, usable, mode, notes,
                           effective=(window if can_crop else vggt_win),
                           limb=_mask_bbox(leg), verdict=verdict))

        offset_from_centre_px = np.hypot((left + right) / 2 - frame_width / 2,
                                         (top + bottom) / 2 - frame_height / 2)

        records.append({
            "index": frame_index + 1, "source": os.path.basename(path),
            "overlay": f"{frame_index:02d}_{os.path.splitext(os.path.basename(path))[0]}.png",
            "reasons": notes, "output": os.path.basename(out_path),
            "frame_size": [frame_width, frame_height],
            "window": [left, top, right, bottom],
            "cube_bbox": None if cube is None else cube.tolist(),
            "band_bbox": None if band is None else band.tolist(),
            "clipped": bool(clipped),
            "offset_px": float(offset_from_centre_px), "mode": mode,
            "cube_ok": cube_ok, "band_ok": band_ok,
            "cube_seen": cube_seen, "band_seen": band_seen,
            "severity": severity, "verdict": verdict,
        })
        label = {PASS: "PASS", WARNING: "WARN", REJECT: "REJECT"}[verdict]
        print(f"  {os.path.basename(path):<16} -> {os.path.basename(out_path)}  "
              f"{mode:<13} {right-left}px ({(right-left)/min(frame_width,frame_height):.0%} of frame)  {label}"
              f"{('  [' + severity + ']') if severity else ''}"
              f"{('  ' + '; '.join(notes)) if notes else ''}")

    marker_colour = _average_band_colour(band_colours, total_frames=len(records))

    # If the capture-level colour was discarded for want of corroboration, then
    # as far as the pipeline is concerned this capture has no band — and the one
    # or two frames that thought they saw one must not still be reported as
    # clean. Leaving them PASS tells a reviewer the band was fine on that frame,
    # which is the opposite of what was just decided.
    if marker_colour is None:
        for record in records:
            if record.get("band_seen"):
                record["band_seen"] = False
                record["band_bbox"] = None
                record["verdict"], record["reasons"], record["severity"] = (
                    _frame_verdict(record["cube_seen"], False,
                                   record["cube_ok"], True))

    if marker_colour:
        print(f"\n  Marker colour learned from {len(band_colours)} frames: "
              f"RGB {[int(v) for v in marker_colour['rgb']]}, "
              f"excess-green {marker_colour['exg']:+.0f} "
              f"— handed to Stage 3 instead of a fixed threshold")

    # A warning is a usable frame. Only a reject is withheld, which is the whole
    # point of having three verdicts rather than two: a capture with no marker
    # band is measurable, it simply cannot be cut without a person placing the
    # plane, and refusing it outright was costing captures that were fine.
    bad = [r for r in records if r["verdict"] == REJECT]
    warned = [r for r in records if r["verdict"] == WARNING]
    good = [r for r in records if r["verdict"] != REJECT]

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
        "all_passed": not bad and not warned and len(good) >= min_frames,
        "usable": not bad and len(good) >= min_frames,
        "warned": [r["source"] for r in warned],
        "rejected": [r["source"] for r in bad],
        "marker_colour": marker_colour,
        "frames": [{
            "index": r["index"],
            "source": r["source"],
            # `verdict` is the real answer; `accepted` is kept as the boolean
            # older consumers read, and now means "used by the pipeline" —
            # which includes WARNING frames.
            "verdict": r["verdict"],
            "accepted": r["verdict"] != REJECT,
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

    passed = [r for r in records if r["verdict"] == PASS]
    print()
    print("=" * 60)
    print(f"STAGE 0 VERDICT: {len(passed)} pass, {len(warned)} warning, "
          f"{len(bad)} reject  (of {len(records)} submitted)")
    print("=" * 60)
    if warned:
        print("  Warnings — these frames are used, with a caveat:")
        for r in warned:
            print(f"    img{r['index']:<3} {r['source']:<18} {'; '.join(r['reasons'])}")
    if bad:
        print("  Rejected — these frames are not used:")
        for r in bad:
            print(f"    img{r['index']:<3} {r['source']:<18} {'; '.join(r['reasons'])}")
    if not warned and not bad:
        print("  Every frame framed cleanly.")

    if strict and (bad or len(good) < min_frames):
        print()
        print("=" * 60)
        print("STAGE 0: INPUT NOT ACCEPTED — please re-take and re-submit")
        print("=" * 60)
        print(f"  {len(good)} of {len(records)} frames usable; "
              f"{min_frames} are required. Warnings are allowed through — only "
              f"rejects are not.")
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
