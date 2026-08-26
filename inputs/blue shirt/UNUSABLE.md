# blue shirt — not usable

Excluded from results on 2026-08-27, on inspection of the Stage 1/2 output.
The reconstruction itself is wrong, so nothing downstream of it can be
corrected by tuning: the fault is in VGGT's geometry, not in the cut, the
calibration or the mesh.

Do not quote a volume from this capture. It is kept for the record, and
because it is a useful example of a failure the pipeline does **not**
currently detect.

## What the pipeline reported anyway

| | |
|---|---|
| reported volume | 5280 cm³ |
| measured displacement | 3420 cm³ |
| error | **+54.4%** |
| Stage 0 | 0 pass, 6 warning, 1 reject |
| marker colour | discarded — band found on 4 of 7 frames, 5 needed |
| cutting planes | none, so the whole limb was measured |

Every gate that could fire did fire, and the run still produced a confident
number. That is the part worth keeping: the framing gate, the corroboration
rule and the deferred cut all behaved correctly and none of them is a check on
the reconstruction.

## Contributing capture problems

- **Duplicate frames.** The folder holds 13 files that are 7 unique
  photographs plus 6 byte-identical `(1)` copies. A duplicate gives VGGT the
  same viewpoint twice, which adds no parallax while counting toward the frame
  budget. Runs used a deduplicated copy; the duplicates are left here as
  received.
- **The reference cube is occluded by the foot** in most frames, and one frame
  contains neither cube nor marker (`2220050758C1E08767391E403F033669BD71ECAF.jpg`,
  rejected `nothing detected`).
- **No marker band survives the size guard** on any frame — the detector
  returned boxes 1.5–2.0× the limb's own mask area, so there is no cut to
  place.

## If this subject is re-shot

Orbit further round so the cube is never behind the foot, keep the cube fully
in frame in every shot, and use a saturated band colour. Then this becomes an
ordinary capture rather than an excluded one.
