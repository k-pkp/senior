# MLS on the reference cube: plane fit vs quadratic

## Question

`mls_project` fits a degree-2 height field over the local tangent plane. That
exists to preserve curvature on a limb. The reference cube has none — its faces
are planar, and that is known about the object rather than inferred from the
cloud.

On a flat face the quadratic's `a²`, `ab`, `b²` terms have nothing legitimate to
fit, so they can only follow noise and the ghost sheet itself — the very
structure MLS is meant to collapse. A plane cannot do that, so it should flatten
both sheets harder. And Stage 6 measures the cube by fitting planes to its faces,
so flatter faces should feed straight into a better scale.

The predicted cost is at the edges: a neighbourhood spanning two faces gets a
plane sitting diagonally across both, pulling the corner inward. Rim rounding is
already 0.11–0.13 cm and this can only add to it.

## Method

`MLS_BOX_POLYNOMIAL` added to `pipeline/config.py`, threaded through
`pipeline/stages/clean.py` so the reference and the limb can differ (MLS already
ran per cluster). Stages 3–6 run twice from the same Stage 1–2 source
(`--src rerun_leg`), so the only difference is the fit.

    python stagerun.py 3-6 -i ./inputs/small_leg --name mls_quad  --src rerun_leg
    python stagerun.py 3-6 -i ./inputs/small_leg --name mls_plane --src rerun_leg

## Result

```
                          quadratic     plane        verdict
MLS moved p95              0.00424     0.00352      -17%, smooths harder as predicted
horizontal disagree          1.11%       1.02%      better
vertical deficit            +1.32%      +1.27%      better
squareness deviation  -0.30/-0.04/+0.33  -0.28/-0.03/+0.31   better
face RMS                   3.660 mm    3.952 mm     WORSE
face p95                  15.850 mm   15.934 mm     WORSE
reference volume         2692.9 cm³  2688.4 cm³     -4.5, further from 2744
limb volume              1074.08     1080.90        +6.8
box mesh faces             14,648      20,828       +42%
alpha closure                 55x         55x       unchanged
```

## Reading

The mechanism works as far as it goes: the plane fit does smooth harder — p95
movement fell 17%, and 32% more points ended up within a fixed band of the fitted
face plane (4,912 against 3,718).

But it does not make the face flatter, and the aggregate metrics were misleading
about why. Plotting a single face edge-on — deviation from its own fitted plane
against position along the face — settles it:

```
interior flatness (rims excluded)   quad 2.02 mm sd   plane 2.04 mm sd
whole face                          quad 2.83 mm sd   plane 2.78 mm sd
```

Effectively identical. The earlier "face RMS got worse" reading was an artefact
of a metric that sampled a band near each face plane and therefore caught the
extra corner geometry.

What the profile does show is more important than the question being asked. **The
face is bowed by roughly 3–4 mm peak to peak** — the quadratic run sweeps from
−2 mm to +1.5 mm across the face, the plane run from +1.5 mm to −0.5 mm, tilting
opposite ways. A cube face is physically flat to a fraction of a millimetre, so
that bow is a real geometric error, and it is **larger than the ghost separation
MLS exists to collapse and larger than any difference between the two fits**.

That explains why the fit type barely matters: MLS fits a *local* neighbourhood,
so if the whole face is curved, the local fit follows the curve. Neither a plane
nor a quadratic removes a bow that is present in the input.

The rims behave as predicted: outside the interior cutoff both collapse to −6 to
−8 mm, steeper and earlier in the plane run, which is the edge damage from a
neighbourhood spanning two faces.

## Decision

Not adopted. `MLS_BOX_POLYNOMIAL = True` — unchanged behaviour, confirmed by the
face profile showing no flatness benefit to trade the rim cost against. The flag
is kept so the experiment can be repeated.

## What this actually surfaced

The reference cube's faces bow by 3–4 mm. That is the dominant geometric error on
the object that sets the scale for every reported volume, it is not addressed by
anything in the current pipeline, and it is a plausible common cause behind three
open items that have resisted the framing work: the marker size spread (5–7%), the
vertical deficit, and the reference residual.

Where the bow comes from is untested. Candidates worth separating: VGGT's own
depth curvature across a planar surface; MLS following a bow it inherited rather
than creating one (testable by measuring the same face before MLS); or the
alpha-shape surface bulging between sample points.

## Figures

- `mls_box_quad_vs_plane_overview.png` — top view and a mid-height slab, both runs.
  The two clouds are near-identical, which is why the aggregate numbers moved so
  little.
- `mls_box_quad_vs_plane_face_profile.png` — one vertical face edge-on. This is
  the figure that matters: the bow and the rim collapse are both visible directly.

## Raw output

`mls_plane_vs_quadratic.log` beside this file. Full runs preserved in
`work/mls_quad/` and `work/mls_plane/`.
