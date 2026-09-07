# Measuring Ground Truth

The pipeline reports a volume. Nothing in it can tell you whether that volume is
right. Only a physical measurement of the same limb can, and this document is
how to take one so that a disagreement means something.

The short version: **measure circumferences with a tape and sum truncated
cones.** Keep water displacement as a cross-check, and only with an overflow
spout. Both methods and the reasoning behind that order are below.

---

## What has to be true of a ground truth

It has to be **the same quantity the pipeline reports.** The pipeline measures
either the limb below one marker band or the segment between two of them, and
those differ by more than a litre. A truth measured foot-to-band compared
against a run cut band-to-band is not a 30% error, it is two different numbers.

It has to be **repeatable.** A method whose run-to-run spread is larger than the
error being investigated cannot resolve that error, however careful the
arithmetic afterwards.

It has to have a **known direction of bias.** Every method is biased. A bias you
can name and size is a correction; a bias you cannot is an unknown that
contaminates every comparison drawn against it.

---

## Method 1 — circumferences and the disc model (preferred)

This is the standard clinical alternative to displacement volumetry and the one
to use here.

### Procedure

1. Mark the limb at **4 cm intervals** from the lower band up to the upper band.
   Mark the skin, so the same heights can be found again.
2. Measure the circumference at each mark with a tape. Same tape, same tension,
   tape flat against the skin without compressing it.
3. Record every circumference with its height above the lower band.
4. Repeat the whole set three times. Keep all three.

### The arithmetic

Each 4 cm slice is treated as a truncated cone between the two circumferences
bounding it:

    V_slice = h × (C₁² + C₁·C₂ + C₂²) / (12π)

with `h` the spacing and `C₁`, `C₂` the circumferences at its ends. The segment
volume is the sum of the slices. `tools/limb_volume_from_tape.py` does this from
a CSV, in the same units and column layout the comparison scripts expect.

### What it gets wrong

The model treats each cross-section as a circle. A calf is elliptical, and a
circle of the same circumference encloses more area than an ellipse does, so the
model **reads high**. The size of that bias depends on how elliptical the limb
is: at an axis ratio of 1.3, which is what the pipeline measures at these
subjects' ankles, the overestimate is about 2%.

That bias is constant across subjects and heights, which is what makes it
usable. A constant bias does not stop you comparing captures with each other,
and it can be stated alongside the result rather than hidden in it.

### Why it is preferred

Every error the displacement method carries — fill level, film, tilt, a shifting
tub — is absent, because nothing is immersed. Tape measurements repeat to about
1%.

It also produces far more than a volume. Seven circumferences up the limb are a
**profile**, and the pipeline reports circumference at any height too, so the two
can be compared point by point. A single volume lumps every error together; a
profile says where the error is. On 2026-09-04 the two band girths already in
hand were what localised a reconstruction problem to the upper band — seven
points would have localised it properly.

---

## Method 2 — water displacement (cross-check only)

Archimedes: a limb lowered into a full container displaces its own submerged
volume, and that water can be weighed. One gram of water is 1.00 mL to within
0.3% at room temperature, so a scale is a volumetric instrument here.

### Procedure

1. Use a container with an **overflow spout** at a fixed height — a tube or a
   notch, not a rim.
2. Fill until the spout **stops dripping on its own**. This is the only way to
   put the starting level at a defined point, and it is the step that matters
   most.
3. Put the catch vessel on the scale and tare it.
4. Lower the limb until the water surface sits **at the marker band**. Hold it
   still; muscle contraction changes limb volume.
5. Wait for the spout to stop dripping.
6. Weigh the caught water. That mass in grams is the displaced volume in cm³.

For a segment between two bands, measure to the upper band and to the lower band
separately; the segment is the difference.

### Why the catch vessel, and not the tub

Weighing the tub before and after is the obvious method and it has a bias that
the catch vessel does not. Removing the limb carries a film of water out with
it, and re-weighing the tub counts that film as displaced volume:

    W₁ − W₂  =  V_displaced  +  V_film

A wet calf carries perhaps 20–50 g. That is 1–2% of a 2100 cm³ measurement to
the upper band, but **7–17% of a 300 cm³ measurement to the ankle** — which is
why the below-lower-band figures in `inputs/groundtruth0-6.csv` are the least
trustworthy column in it.

Catching the overflow avoids this entirely: the spill happens during immersion,
and the film leaves afterwards.

### The error that dominates without a spout

Water only leaves once the level reaches the overflow point. Start below it and
the limb's first displacement merely raises the level, spilling nothing — that
volume is invisible to the scale, and the measurement reads low.

A container wide enough for a leg has roughly a 1000 cm² water surface, so:

| shortfall below the overflow point | volume never spilled |
|---|---|
| 1 mm | 100 cm³ |
| 5 mm | 500 cm³ |
| 8 mm | 800 cm³ |

Against a 2100 cm³ reading, 5 mm is a −24% error. Without a spout, 5 mm is easy
to be off by: surface tension lets water stand proud of a rim before it breaks,
then it runs and settles somewhere below. The starting point is both unknown and
different every time.

**This is the leading candidate for why `inputs/groundtruth0-6.csv` reads low
against the pipeline**, and it has not yet been separated from the pipeline's own
over-reading. See "Open question" below.

### Scale tilt

A load cell reads the force component along its own axis, so a tilted scale
scales both weighings — and their difference — by roughly cos θ:

| tilt | reading error |
|---|---|
| 5° | −0.4% |
| 10° | −1.5% |
| 20° | −6.0% |
| 41° | −25% |

Real, worth removing, but far too small to explain a 25–35% discrepancy unless
the tilt is extreme enough to be obvious. Level the scale and stop thinking
about it.

### Two checks worth running once

**Weigh a known volume.** Pour a measured 2000 mL from a graduated flask into the
catch vessel. If the scale reads 2000 g, the scale and its levelling are fine and
neither is a suspect any more. Two minutes, and it removes a whole branch of the
error tree.

**Repeat one limb three times without changing anything.** Close agreement means
the setup is stable and any remaining error is a constant bias. Scatter means
something moves between measurements — a shifting tub, an inconsistent fill —
and that has to be fixed before the numbers mean anything.

---

## What to record

One row per limb, so the comparison scripts can read it directly:

| column | meaning |
|---|---|
| `no` | subject number |
| `sex` | `M` or `W` |
| `<side>_low_delta` | displaced volume to the **lower** band, cm³ |
| `<side>_up_delta` | displaced volume to the **upper** band, cm³ |
| `<side>_vol` | segment between the bands, cm³ — `up_delta − low_delta` |
| `band_<side>_lower` | taped girth at the lower band, cm |
| `band_<side>_upper` | taped girth at the upper band, cm |
| `band_<side>_separation` | **tape distance between the two bands, cm** |
| `<side>_tape_vol` | disc-model volume between the bands, cm³ |

Two of these are new and both earn their place.

`band_<side>_separation` is the one measurement that lets plane placement be
checked without trusting the reconstruction: the pipeline reports where it cut,
and a tape says how far apart the bands actually are. Without it, a cut in the
wrong place and a limb reconstructed the wrong size look identical in the volume.

`<side>_tape_vol` carries the disc-model result alongside the displacement one,
so the two methods can be compared on the same limb rather than one silently
replacing the other.

Record the full circumference profile too — every height, every repeat — in a
separate file. The summary row throws away the part that localises error.

---

## Open question, as of 2026-09-04

Fourteen captures were measured against `inputs/groundtruth0-6.csv` and the
pipeline read **+24.6% to +35.8% high** on every capture that produced a sound
two-plane cut. Two mechanisms have been identified, and they point in opposite
directions:

- **The pipeline over-reads girth where the reconstruction ring is incomplete.**
  Circumference is an ellipse fitted to a ring of surface points, and where the
  back of the calf did not reconstruct the fit extrapolates across the gap.
  Measured: 94% ring coverage gives +5.3% girth error, 26% coverage gives
  +46.6%. Volume goes as girth squared, so +11% girth is roughly +23% volume.

- **The displacement method under-reads if the container is not at its overflow
  point**, by 100 cm³ per millimetre of shortfall, as above.

Neither has been isolated. Until one of them is, the difference between the two
columns cannot be attributed to either side, and quoting it as a pipeline error
overstates what is known.

The tape profile settles it. It shares no error mechanism with either — no water,
no reconstruction — so it can adjudicate between them at every height rather than
only at the end.
