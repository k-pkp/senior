"""Pipeline-wide configuration constants."""

# Stage 6 volume calibration
# The ArUco marker is identified as the "obj" cluster (non-box).
# REFERENCE_REAL_SIZE_CM is the real-world linear size of the ArUco marker cube.
REFERENCE_REAL_SIZE_CM = 14.0     # real linear size of reference in cm

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

# Reject marker planes found in the bottom fraction of the object's height.
# Feet, shadows under the arch and the floor junction all live there, and a cut
# line placed that low would discard nearly the whole limb. Set to 0 to disable.
MARKER_MIN_HEIGHT_FRAC = 0.20

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
