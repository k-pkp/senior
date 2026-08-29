"""Pipeline-wide configuration constants."""

import os

# Stage 6 volume calibration
# The ArUco marker is identified as the "obj" cluster (non-box).
# REFERENCE_REAL_SIZE_CM is the real-world linear size of the ArUco marker cube.
#
# There are TWO physical cubes, and using the wrong one rescales every reported
# volume by (a/b)**3 with no visible sign:
#
#     10.0 cm   3D-printed cube      captures from Aug 2026 onward
#                                    (champ, sunshine, keng, *_shirt)
#     14.0 cm   handmade cardboard   the original fixtures
#                                    (small_leg, short_leg, est_325)
#
# The default follows the current hardware. Override per run for the older
# captures rather than editing this line:
#
#     REFERENCE_REAL_SIZE_CM=14 python stagerun.py 6 --name small_leg
#
# Neither value has been verified with calipers. A 2 mm build error is 2.0%
# linear on the 10 cm cube = 6.1% of volume on every result, which is larger
# than it was on the 14 cm one (1.4% linear, 4.3% volume).
REFERENCE_REAL_SIZE_CM = float(os.environ.get("REFERENCE_REAL_SIZE_CM", 10.0))

# Stage 2 multi-view consistency
#
# Keep only points whose depth is corroborated by at least this many OTHER
# views. 0 disables it.
#
# DISABLED, and the reason matters: this was built to remove ghost sheets and
# it does not work. Measured on small_leg, local shell thickness was unchanged
# at every setting while up to 41% of the cloud was discarded:
#
#     min_views=0   527,769 pts   shell std 0.72 mm
#     min_views=2   446,218 pts   shell std 0.73 mm
#     min_views=3   311,760 pts   shell std 0.72 mm
#
# Multi-view consistency assumes each view's errors are independent, so a wrong
# point fails corroboration. The ghost sheet is the same model making the same
# mistake in every view — the views agree with each other and the ghost passes.
# At min_views=3 the thinned cloud also destabilised Stage 3 clustering.
#
# Kept because it is a genuine measure of geometric self-consistency and useful
# as a Stage 1 diagnostic; just not a ghost filter.
MULTIVIEW_MIN_VIEWS = 0
MULTIVIEW_REL_THRESHOLD = 0.05

# Stage 3 floor removal band, in multiples of the RANSAC distance threshold.
#
# VGGT reconstructs the floor as a duplicate pair of sheets — the same ghosting
# it produces on the limb — so the floor is about two thresholds thick. Removing
# only the RANSAC inliers takes the middle and leaves BOTH skins behind:
#
#   distance from fitted plane, points surviving a 1x removal
#     [-0.010,-0.005)  20,080
#     [-0.005,+0.005)     614      <- the band that was removed
#     [+0.005,+0.010)  21,669
#
# Those 41,749 leftover points are a full-size slab of floor. DBSCAN then links
# the limb, the cube and that slab into one cluster, so Stage 3 exported the
# whole scene as the limb and labelled a 3,743-point sliver as the object.
#
# At 2x the floor stops reappearing (the next dominant plane is a wall, dot 0.19
# with the floor instead of 1.00) and the box cluster becomes cube-shaped
# (cubeness 0.87, extent 0.317/0.329/0.365). Past 2x the gain flattens — 3x
# removes only 7k more, 4x only 3k — while eating into whatever rests on
# the floor.
PLANE_REMOVAL_BAND_MULT = 2.0

# Stage 3 marker detection
#
# The old rule was `saturation > 15 AND hue > 60`, which accepts hue 60-360 —
# everything except red, orange and yellow — with no brightness floor at all.
# On inputs/small_leg that classified two things as markers:
#
#   shadow  RGB(8,6,8)      V=3.1%    hue is arbitrary at that brightness
#   skin    RGB(139,87,89)  hue=358degrees, which `hue > 60` happily admits
#
# The shadow cluster had 3349 supporting points against 197 for the real band,
# so it won the cut and the pipeline measured a slab of ankle.
#
# These bounds describe a genuinely green marker. MARKER_VAL_MIN is the
# important one: hue is numerically unstable as value approaches zero, so any
# dark pixel can land anywhere on the wheel.
MARKER_HUE_MIN = 70.0     # degrees; below this is yellow/orange/red
MARKER_HUE_MAX = 180.0    # degrees; above this is cyan/blue/magenta
MARKER_SAT_MIN = 25.0     # percent; washed-out pixels have no reliable hue
MARKER_VAL_MIN = 15.0     # percent; below this hue is noise
# 2G-R-B. This is the rule that actually finds the band, not the hue window.
# The small_leg marker is khaki, RGB(60,52,30) — hue 44 degrees, so it fails
# every green hue test — but ExG +14 against skin at -54 separates it cleanly.
# Raising this to 15 silently lost the only real marker in the dataset.
MARKER_EXG_MIN = 10

# Reject marker planes found low on the object. Feet, shadows under the arch
# and the floor junction all live there, and a cut line placed that low would
# discard nearly the whole limb.
#
# The floor is expressed in REFERENCE CUBE HEIGHTS above the ground, not as a
# fraction of the limb's own span, because the span is not a property of the
# subject -- it is however much leg happened to be in shot. On inputs/small_leg
# the cluster is a calf and a foot; on the Aug 2026 captures it runs from the
# floor to mid-thigh, so the same physical ankle band sits at 44% of the span
# in one and 18% in the other. A fraction rule cannot hold both. The cube is a
# known 10 cm standing on the same floor, so one cube height is one physical
# length that transfers between captures.
#
# Measured on the Aug 2026 captures: the cube is 15-17% of the limb span, and
# the genuine ankle bands sit at 18.2% (champ) and 18.3% (black shirt) -- above
# one cube height, below the old 20% fraction, and rejected by it. The false
# planes it exists to catch are the floor junction at 6.8% and the arch at 7%,
# well under one cube height.
MARKER_MIN_HEIGHT_CUBES = 1.0

# Which of the validated marker planes the cut actually uses.
#
#     "upper"  keep everything BELOW the highest valid plane
#     "span"   keep what lies BETWEEN the lowest and highest valid planes
#     "auto"   follow the bands: one band -> upper, two bands -> span
#
# What "auto" reads is deliberately not one detector's opinion. Stage 0 counts
# the bands in the photographs, across frames, and Stage 3 fits planes to the
# marker points in 3D; auto cuts a span only when BOTH say two. Either alone is
# a known failure: Stage 3 has fitted a plausible second plane to clothing and
# to a floor junction, and Stage 0's detector will return a second box for a
# cord's own shadow. Requiring agreement means a single false positive can no
# longer change what quantity is being reported.
#
# It still is not a measurement. What a run reports has to match what the ruler
# measured, and the Aug 2026 displacement volumes were taken foot-to-upper-band
# -- so a two-band capture measured against those must be run with an explicit
# --cut-mode upper, and re-measuring the ground truth is the alternative. Auto
# picks the reading that matches the bands the subject is wearing, which is the
# right default for a new capture and the wrong one for an old comparison.
# Every run prints which mode it resolved to and why.
MARKER_CUT_MODE = "auto"

# Fallback for MARKER_MIN_HEIGHT_CUBES when no reference cube was segmented.
# Set either to 0 to disable the gate.
MARKER_MIN_HEIGHT_FRAC = 0.20

# Upper bound on the band/limb contrast score. See marker_mask_by_contrast.
#
# The score is 0 at the limb's measured colour and 1 at the band's, so a real
# marker sits near 1. Anything well above 1 is FURTHER from the limb than the
# band is, along the band's own axis -- which no marker on that limb can be.
# The rule had a floor and no ceiling, and that is what let it select clothing
# and floor tile. Measured on inputs/champ, using its own learned colours:
#
#     skin           -0.06     correctly rejected
#     floor tile     +0.94     selected
#     the band       +1.00     by definition
#     neutral grey   +1.14     selected
#     grey shorts    +1.54     selected -- 5,265 points, and it won the cut
#
# 1.5 keeps a full half-width of slack above the band for shading and view
# angle while excluding the shorts. On inputs/small_leg nothing between 0.5
# and 1.5 changes, because its axis is long enough that neutral surfaces score
# -0.30 and never approached the window.
MARKER_SCORE_MAX = 1.5

# Shortest usable band-vs-limb separation, in chromaticity.
#
# The score divides by |axis|^2, so a short axis magnifies every score and the
# window above stops separating anything. Measured |axis|, with what a neutral
# grey then scores:
#
#     small_leg     0.0941   -0.30   works
#     black shirt   0.0433   +1.12   grey selected
#     champ         0.0323   +1.14   grey selected
#     sunshine      0.0308   +2.62   the whole limb selected
#     keng          0.0213   +0.04   marginal
#
# Below this the learned colour is refused and detection falls back to the
# hand-tuned config window, which is what inputs/orange_shirt already does --
# and orange_shirt is the most accurate capture in the set. Refusing loudly
# beats measuring through a discriminant that cannot discriminate.
MARKER_MIN_AXIS = 0.05

# Maximum angle, in degrees, between a marker plane's normal and the limb's own
# axis at that height.
#
# A cord tied round a limb lies perpendicular to it, so the plane's normal
# should point along the limb. This is a shape test rather than a colour one,
# and it is the only gate that catches a plane fitted to a large blob of skin:
# such a plane takes the blob's own principal direction, which has nothing to
# do with the limb. Measured, against each capture's own limb axis:
#
#     black shirt band   17.3 deg from vertical   -> valid, and +4.4% vs truth
#     keng band          15.7 deg                 -> valid, and +1.9% vs truth
#     keng false plane   83.1 deg                 -> rejected
#     sunshine, only     87.4 deg, 43,468 pts     -> rejected
#     champ shorts       41.5 deg,  5,265 pts     -> rejected
#
# 35 deg is generous: it admits every genuine band measured so far with at
# least 12 degrees to spare, and excludes every false one by at least 6.
MARKER_MAX_AXIS_ANGLE_DEG = 35.0

# Minimum points in a marker cluster before a plane is fitted to it.
#
# Was hardcoded at 150 in compute_cluster_planes. A real marker band is small:
# on small_leg the band is 99 points out of 182k, because only the part facing
# a camera reconstructs. 150 rejected it. The old loose colour rule cleared the
# bar only by padding clusters with shadow, which is what produced the
# 3349-point "marker" on the ankle.
MARKER_MIN_CLUSTER_PTS = 40

# Bottom-cap grid spacing, as a multiple of scan point spacing.
#
# The cap tiles a flat cross-section, so it does not need surface-resolution
# sampling — it only needs to be dense enough that the alpha shape cannot punch
# through it. The smallest alpha multiplier is 8x spacing (see
# workers/recons_methods_worker.py), so 3x leaves a wide margin.
#
# At 1.0 the cap was 100,559 points on small_leg — 77% of the whole cut cloud
# was fabricated floor, and it fed Stage 4 a dense coplanar slab, which is the
# degenerate input for Delaunay tetrahedralization. This constant was implicitly
# 1.0 before Stage 3 stopped downsampling, when scan spacing was ~4x coarser.
CAP_SPACING_MULT = 3.0

# Bounds on the resulting cap point count, applied after the multiplier.
#
# The multiplier alone is not safe at both ends. The reference cube's bottom
# cross-section is tiny (its underside is against the floor and barely
# reconstructs), so 3x took its cap from 150 points to 16 — too few to close
# anything, and the alpha search had to run out to 70x before the box sealed.
# The limb's foot has ~200x that area and needed the coarsening.
#
# So: pick spacing from the multiplier, then adjust it until the count lands in
# this window. Area per point is spacing squared, so spacing = sqrt(area / n).
CAP_MIN_PTS = 200
CAP_MAX_PTS = 20000

# Stage 3 MLS surface projection
#
# Collapses VGGT's duplicate ghost sheet onto a single surface by projecting
# each point onto a locally fitted quadratic. This is the only thing that has
# worked on the ghosting — filtering cannot, because the ghost is parallel to
# the true surface (invisible to the normal filter) and repeated consistently
# across views (invisible to multi-view consistency).
#
# Radius is in multiples of point spacing and MUST exceed the ghost separation,
# or both sheets never share a neighbourhood: at 2.0 nothing moved at all.
#
# It moves points rather than removing them, so it costs some volume — measured
# on small_leg (shell std / convex hull):
#     off    0.93 mm   1922 cm3
#     3.0x   0.60 mm   1887 cm3   -1.8%
#     4.0x   0.41 mm   1833 cm3   -4.6%
#     6.0x   0.29 mm   1788 cm3   -7.0%
# Whether that shrinkage is noise removal or real surface is unresolved: it
# depends on whether the true surface is the inner or outer sheet, which we
# cannot currently determine. 0 disables.
MLS_RADIUS_MULT = 4.0

# Fit a plane rather than a quadratic when smoothing the reference cube.
#
# The quadratic exists to preserve curvature on a limb. The reference has none:
# its faces are planar, and that is known about the object rather than inferred
# from the cloud. On a flat face the quadratic's a^2, ab and b^2 terms can only
# fit noise and the ghost sheet itself, so it can curve to follow the very
# structure MLS is meant to collapse. A plane cannot, so it flattens both sheets
# harder -- and Stage 6 measures the cube by fitting planes to those faces, so
# flatter faces feed straight into the scale.
#
# The cost is at the edges: a neighbourhood spanning two faces gets a plane
# sitting diagonally across both, which pulls the corner inward. Rim rounding is
# already 0.11-0.13 cm and this can only add to it. Whether the trade is
# favourable depends on whether flat interiors outweigh sharper rims.
MLS_BOX_POLYNOMIAL = True

# Stage 2/3 ghost reduction
#
# Voxel size for ghost dedup = GHOST_VOXEL_FACTOR * mean nearest-neighbour dist.
# This is the pipeline's dominant decimation step (~97% of points discarded at
# this stage alone), so it controls final mesh density more than anything else.
# Lower keeps more surface detail but retains more VGGT ghost artifacts.
# Points below are the combined total across the box and limb clusters, since
# the dedup runs per cluster:
#   1.5  -> ~13k pts   (original, faceted meshes)
#   0.75 -> ~26k
#   0.65 -> current
#   0.5  -> ~59k
#   0.35 -> ~120k
#   0    -> dedup disabled entirely; every point kept, ghost layers survive
GHOST_VOXEL_FACTOR = 0.65

# Stage 1 frame limits
DEFAULT_MAX_FRAMES_MPS = 6

# Model weights
#
# VGGT ships two checkpoints under different terms:
#   facebook/VGGT-1B              CC BY-NC-SA 4.0 — non-commercial only, ungated
#   facebook/VGGT-1B-Commercial   vggt-aup-license — commercial OK, gated
#
# The commercial repo is gated, so it needs an authenticated download: accept the
# terms on the model page, then export HF_TOKEN (or run `huggingface-cli login`).
# Set VGGT_USE_COMMERCIAL = False to fall back to the non-commercial checkpoint.
#
# The commercial licence's Acceptable Use Policy forbids unlicensed medical/health
# professional practice and inferring health data without consent — relevant when
# limb measurements are involved.
VGGT_USE_COMMERCIAL = True
VGGT_COMMERCIAL_REPO = "facebook/VGGT-1B-Commercial"
VGGT_COMMERCIAL_FILE = "vggt_1B_commercial.pt"
VGGT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

# Image extensions accepted by the input loader
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".heic", ".heif")

# Stage 6 marker cross-check
#
# The reference cube is the pipeline's only ruler, so it cannot check itself:
# the scale is derived from its own fitted faces, and it therefore measures
# REFERENCE_REAL_SIZE_CM exactly whatever it really is. The squareness check
# above says as much — it cannot see a common-mode error, because inflating all
# three edges equally leaves the shares untouched.
#
# The markers printed on the cube are the missing second structure. ArUco finds
# their corners to sub-pixel accuracy, and those corner pixels index the
# pointmap directly, so a marker lifts into 3D with no colour thresholding, no
# meshing and no image-to-cloud mapping. Measured on both datasets:
#
#     marker / face ratio     small_leg 0.4518   est_325 0.4507
#
# Two independent captures, different scenes and scale factors, agreeing to
# 0.24%. That reproducibility is what makes it usable as a check.
#
# Most of its value needs no constant at all. A printed marker is physically
# flat and square, so departures are pure reconstruction error:
#
#     flatness      0.04-0.45 mm    the surface is locally accurate
#     aspect        1.077-1.079     a square reconstructs ~8% out of square
#     size spread   3.7-3.9%        one physical square, measured on five faces
#
# REFERENCE_MARKER_CM is the black square's outer edge, and it is deliberately
# left unset. The sheet was designed at 7.00 cm but did not print at that size:
# rectifying the photographs through the marker's own homography and holding the
# box at its measured 14.00 cm puts the printed square at 6.49-6.58 cm, roughly
# 93% of design and consistent with "fit to page" scaling. Setting 7.00 here
# would bake in a false 10% discrepancy, so leave it None until the physical
# square is measured with a ruler; the calibration-free checks run regardless.
REFERENCE_MARKER_CM = None

# ArUco family on the reference cube: ids 10-14, one per visible face, detected
# on every frame of both datasets.
REFERENCE_MARKER_DICT = "DICT_5X5_250"

# Warn above these. Both are calibration-free, so they work before
# REFERENCE_MARKER_CM is known.
#   spread  — the same physical square measured on five faces; 3.9% observed on
#             a run whose reference residual was -1.86%, so this is roughly the
#             current state of the art rather than a failure threshold.
#   aspect  — longest over shortest side of a printed square; 1.08 observed.
MARKER_SPREAD_WARN = 0.06
MARKER_ASPECT_WARN = 1.15
