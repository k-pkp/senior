# The whole pipeline, in one chart

Every stage, every sub-process, and the inputs and outputs of each — as a single
diagram rather than the eight scoped ones in
[`pipeline_flowchart.md`](pipeline_flowchart.md). That file is for reading a
stage at a time; this one is for seeing where anything sits in the whole.

It is deliberately large — around 3300 × 20000 pixels when rendered. Open it in
a viewer that renders mermaid and zoom; it is meant to be read a stage at a time
at full magnification, not taken in at once.

**Reading it**

Every box names the function and then states its own input and output:

```
_subject_bounds(cube_box, band_box, limb_mask, band_heights=1.6, pad=0.05)
in  ▸ cube box + band box + limb mask
out ▸ union box (x0,y0,x1,y1)
limb followed up to 1.6 × cube height
```

So a box tells you what to hand it and what comes back, and the **edge label**
tells you what is travelling between two boxes. Nothing has to be looked up
elsewhere to follow the dataflow.

| shape / colour | meaning |
|---|---|
| rounded box, white | a function or a step that transforms data |
| parallelogram, blue | a file written to disk, with its full contents listed |
| hexagon, amber | a decision a **person** makes |
| diamond, grey | a decision the **code** makes |
| red border | a neural network — the only two places a model runs |
| green rounded | an invariant that is enforced, not assumed |
| dashed edge | data that informs a later stage without being its main input |
| ⚠ in a label | a known weakness at that exact step |

**One property to hold onto while reading**: the subject's volume is never
computed before a person confirms the cut. Stages 4-6 appear twice; the first
time they run on the reference cube alone.

---

## The chart

```mermaid
flowchart TB

%% ═══════════════════════ STAGE 0 ═══════════════════════
    PHOTOS[/"photos<br/>6-12 files, JPEG or HEIC<br/>typically 3024 × 4032, 9:16"/]:::file

    subgraph SG0["STAGE 0 · framing gate — pipeline/stages/prep.py · prepare_frames()"]
        direction TB
        P_READ["<b>_read_image_bgr(path)</b><br/>in ▸ file path<br/>out ▸ BGR (H,W,3), or None<br/>cv2.imread, falling back to PIL + pillow_heif —<br/>OpenCV cannot decode HEIC, which phones shoot by default"]
        P_ARUCO["<b>_cube_faces(grey, dict_name)</b><br/>ArUco DICT_5X5_250<br/>in ▸ grey (H,W) uint8<br/>out ▸ list of marker quads, each (4,2) float"]
        P_HOMOG["<b>_face_corners(quad, face_cm, marker_cm)</b><br/>in ▸ one (4,2) marker quad + both known sizes<br/>out ▸ (4,2) corners of the WHOLE cube face<br/>via homography — the marker only samples the face"]
        P_DINO1["<b>_cube_bbox(grey, image_pil)</b><br/>GroundingDINO · prompt 'cardboard box'<br/>in ▸ grey + PIL RGB<br/>out ▸ (x0,y0,x1,y1) float, or None"]:::nn
        P_CUBE["<b>cube box</b> = ArUco faces ∪ detector box<br/>in ▸ face quads (N,4,2) + detector box<br/>out ▸ (x0,y0,x1,y1)<br/>⚠ if the detector raised, this is the faces alone — smaller"]
        P_LEG["<b>_leg_mask(image_pil, cube)</b><br/>GroundingDINO 'leg' → SAM, box-prompted<br/>in ▸ PIL RGB + cube box (to exclude overlaps)<br/>out ▸ mask (H,W) bool + limb box (x0,y0,x1,y1)"]:::nn
        P_BAND["<b>_band_bbox(image_pil, bgr, leg_box, limb_mask)</b><br/>GroundingDINO · prompt 'cord'<br/>in ▸ PIL + BGR + limb box + limb mask<br/>out ▸ (x0,y0,x1,y1) or None<br/>must intersect the limb AND be under 0.35x its area —<br/>without the size guard a band-free capture returns the LEG"]:::nn
        P_COL["<b>trace_band_colour(bgr, box, keep_percentile=40, dilate=3)</b><br/>in ▸ BGR (H,W,3) + band box<br/>out ▸ dict: rgb, bgr, limb_rgb, limb_bgr, hsv, exg, n_px<br/>per-column max-deviation trace, then ±3 rows"]
        P_BOUND["<b>_subject_bounds(cube_box, band_box, limb_mask, band_heights=1.6, pad=0.05)</b><br/>in ▸ cube box + band box + limb mask<br/>out ▸ union box (x0,y0,x1,y1)<br/>limb followed up to 1.6 × cube height"]
        P_WIN["<b>_square_window(union, shape, cube, pad)</b><br/>in ▸ union box + frame shape (H,W)<br/>out ▸ square window (x0,y0,x1,y1)<br/>full frame width; slides only vertically"]
        P_VGGTWIN["<b>_vggt_window(shape, target=518, patch=14)</b><br/>in ▸ frame shape (H,W)<br/>out ▸ the centre-crop window VGGT would take<br/>used only as the gate's second chance"]
        P_FIT{"<b>_box_fits_inside(window, box, margin_frac=0.05)</b><br/>in ▸ window + one box<br/>out ▸ bool<br/>run for the cube and the band separately<br/>a band is OPTIONAL and never blocks cropping"}
        P_REASON["<b>_frame_verdict(cube_seen, band_seen, cube_ok, band_ok)</b><br/>in ▸ four bools<br/>out ▸ (verdict, notes, severity)<br/>verdict is PASS, WARNING or REJECT — worst finding wins"]
        P_OVER["<b>_debug_overlay(img, window, cube, band, ok, mode, notes)</b><br/>in ▸ frame + every box + the verdict<br/>out ▸ annotated PNG, max side 1200 px"]
        P_WRITE["<b>_write_frame_for_vggt(image, window, should_crop, output_size=518, out_path)</b><br/>in ▸ PIL RGB + window + crop flag<br/>out ▸ frame_NN.png, 518×518 — or the frame RAW if uncroppable"]
        P_AVG["<b>_average_band_colour(per_frame_colours, total_frames)</b><br/>in ▸ colour dicts from the frames that saw a band<br/>out ▸ one dict, or NONE if fewer than 0.6 of the<br/>submitted frames agree — one detection is not evidence"]

        P_READ -- "grey" --> P_ARUCO -- "(4,2) quad" --> P_HOMOG -- "(4,2) face" --> P_CUBE
        P_READ -- "grey + PIL" --> P_DINO1 -- "box or None" --> P_CUBE
        P_READ -- "PIL + cube box" --> P_LEG -- "limb box + mask" --> P_BAND -- "band box" --> P_COL
        P_CUBE -- "cube box" --> P_BOUND
        P_LEG -- "limb mask" --> P_BOUND
        P_BAND -- "band box" --> P_BOUND
        P_BOUND -- "union box" --> P_WIN -- "window" --> P_FIT
        P_VGGTWIN -- "fallback window" --> P_FIT
        P_FIT -- "bool per object" --> P_REASON -- "verdict" --> P_OVER
        P_REASON -- "crop or not" --> P_WRITE
        P_COL -- "one colour dict per frame" --> P_AVG
    end

    F0A[/"<b>00_prep/images/frame_NN.png</b><br/>518 × 518 uint8 RGB, one per usable frame"/]:::file
    F0B[/"<b>00_prep/framing.json</b><br/>submitted · accepted · all_passed · required<br/>band_cube_heights · pad_frac · centre_on_subject<br/>marker_colour {rgb, limb_rgb, bgr, hsv, exg, n_frames}<br/>frames[]: source, index, mode, accepted,<br/>cube_seen, band_seen, reasons[], severity, overlay"/]:::file
    F0C[/"<b>00_prep/overlays/*.png</b><br/>annotated frames, ≤1200 px"/]:::file
    F0D[/"<b>learned marker colour</b><br/>rgb + limb_rgb + exg<br/>passed to Stage 3, not written as its own file"/]:::file

    GATE{{"<b>every frame usable?</b><br/>PERSON decides<br/>in ▸ framing.json + the overlays<br/>out ▸ proceed · proceed anyway · re-shoot"}}:::human
    RETAKE[/"re-take the named photos"/]:::file

    PHOTOS -- "6-12 files" --> P_READ
    P_WRITE --> F0A
    P_REASON --> F0B
    P_OVER --> F0C
    P_AVG --> F0D
    F0B --> GATE
    GATE -- "no · re-shoot" --> RETAKE --> PHOTOS

%% ═══════════════════════ STAGE 1 ═══════════════════════
    subgraph SG1["STAGE 1 · inference — pipeline/stages/inference.py · run_inference()"]
        direction TB
        I_W["<b>_load_weights()</b><br/>in ▸ nothing (reads pipeline/config.py + HF token)<br/>out ▸ VGGT-1B model + the checkpoint it chose<br/>gated commercial weights, else a LOUD CC-BY-NC fallback"]
        I_SEL["<b>_select_frames(image_names, max_frames)</b><br/>in ▸ sorted list of frame paths<br/>out ▸ capped list (6 on MPS, auto elsewhere)"]
        I_PRE["<b>load_and_preprocess_images(paths, mode='crop')</b><br/>in ▸ S frame paths<br/>out ▸ float tensor (S,3,518,518)<br/>Stage 0 already emitted squares, so this is a RESIZE<br/>and discards nothing"]
        I_NET["<b>VGGT-1B forward pass</b><br/>in ▸ (S,3,518,518)<br/>out ▸ aggregator tokens → camera / depth / point heads"]:::nn
        I_FREE["<b>free the model</b><br/>in ▸ model on GPU<br/>out ▸ empty CUDA cache — next stage starts clean"]
        I_W --> I_SEL -- "S paths" --> I_PRE -- "(S,3,518,518)" --> I_NET -- "nine arrays" --> I_FREE
    end

    F1[/"<b>01_inference/predictions.npz</b><br/>world_points (S,518,518,3) float32 ▸ USED, the only 3D source<br/>world_points_conf (S,518,518) ▸ USED, Stage 2's threshold<br/>images (S,3,518,518) ▸ USED, colours<br/>depth (S,518,518,1) + depth_conf ▸ unused<br/>world_points_from_depth (S,518,518,3) ▸ unused, measured worse<br/>extrinsic (S,3,4) · intrinsic (S,3,3) · pose_enc ▸ unused here"/]:::file
    F1R[/"<b>01_inference/raw/</b><br/>every head's output as PNG, PLY and JSON<br/>+ input_frames.png"/]:::file

    GATE -- "yes, or 'measure anyway'" --> I_W
    F0A -- "518² frames" --> I_PRE
    I_NET --> F1
    I_NET --> F1R

%% ═══════════════════════ STAGE 2 ═══════════════════════
    subgraph SG2["STAGE 2 · point cloud — pipeline/stages/pointcloud.py · export_ply()"]
        direction TB
        C_PICK{"<b>--prediction_mode</b><br/>in ▸ predictions dict<br/>out ▸ which array is unprojected"}
        C_PM["<b>world_points</b><br/>the pointmap head — DEFAULT<br/>out ▸ (S,518,518,3)"]
        C_DP["<b>world_points_from_depth</b><br/>measured worse on both datasets<br/>not used"]
        C_CONF["<b>confidence filter</b><br/>in ▸ conf (S,518,518) + conf_thres=45<br/>out ▸ bool mask — 45 is a PERCENTILE, not an absolute"]
        C_BG["<b>background masks</b> (both off by default)<br/>in ▸ RGB + --mask_black_bg / --mask_white_bg<br/>out ▸ bool mask"]
        C_COLR["<b>colour assignment</b><br/>in ▸ images (S,3,518,518)<br/>out ▸ (N,3) uint8 — each point takes its own pixel<br/>from its OWN view; never averaged across views"]
        C_SOR["<b>spatial outlier removal</b><br/>in ▸ 885,470 × 3<br/>out ▸ 858,554 × 3 — removes 26,916 silhouette flyers"]
        C_PICK -- "pointmap" --> C_PM --> C_CONF
        C_PICK -- "depth" --> C_DP --> C_CONF
        C_CONF -- "masked (N,3)" --> C_BG --> C_COLR -- "(N,3) + (N,3) uint8" --> C_SOR
    end

    F1 -- "world_points + conf + images" --> C_PICK
    F2[/"<b>02_pointcloud/points.ply</b><br/>858,554 points · XYZ float32 + RGB uint8<br/>extent (1.168, 0.969, 1.160) mesh units"/]:::file
    C_SOR --> F2

%% ═══════════════════════ STAGE 3 · pass 1 ═══════════════════════
    subgraph SG3["STAGE 3 --no-cut · segment and detect — pipeline/stages/clean.py · clean_and_extract()"]
        direction TB

        subgraph SG3L["load and thin"]
            L_READ["<b>_load_and_thin(dense_ply)</b><br/>in ▸ points.ply path<br/>out ▸ Open3D PointCloud"]
            L_SOR["<b>remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)</b><br/>in ▸ 858,554 pts<br/>out ▸ 823,550 pts"]
            L_VOX["<b>voxel_down_sample(0.002)</b><br/>in ▸ 823,550 pts<br/>out ▸ 445,654 pts<br/>a SPEED setting — surface decimation happens later at 0.005"]
            L_READ --> L_SOR --> L_VOX
        end

        subgraph SG3A["PHASE A · original VGGT space, BEFORE levelling"]
            A_PLANE["<b>remove_dominant_plane()</b> · RANSAC · core/plane.py<br/>in ▸ 445,654 pts, auto threshold 0.004699<br/>out ▸ 225,808 pts — floor of 219,846 (49.3%) removed<br/>without this the floor connects cube to limb"]
            A_DB["<b>detect_top_k_objects(k=2)</b> · DBSCAN · core/cluster.py<br/>in ▸ 225,808 pts, adaptive eps 0.00628<br/>out ▸ 11 clusters, ranked"]
            A_CUBE["<b>cubeness = min_extent / max_extent</b><br/>in ▸ clusters<br/>out ▸ which is the reference<br/>cluster 1: 0.847 → BOX · cluster 2: 0.423 → OBJ"]
            A_BOXC[/"<b>box cluster</b><br/>106,832 pts, extent (0.319,0.347,0.377)"/]:::file
            A_OBJC[/"<b>object cluster</b><br/>114,282 pts, extent (0.404,0.626,0.265)"/]:::file
            A_MARK["<b>_detect_marker_planes(cluster, height_axis, marker_colour)</b><br/>in ▸ dense limb cluster + learned colour<br/>out ▸ [{centroid (3,), normal (3,), npts}]<br/>chromaticity discriminant: band vs limb CONTRAST"]
            A_UP["<b>restrict to the upper 60% of the span</b><br/>in ▸ candidate band points<br/>out ▸ subset — the foot is wider than the calf"]
            A_FIT["<b>RANSAC plane fit</b><br/>in ▸ 296 selected band points<br/>out ▸ centroid + unit normal"]
            A_SCALE["<b>_ghost_filter_scales(dense_cloud)</b> · ghost.compute_voxel_size<br/>in ▸ cloud<br/>out ▸ voxel_size 0.0050 = 0.65 × mean NN spacing<br/>shared by both clusters so they decimate identically"]
            A_DEDUP["<b>ghost_voxel_downsample(points, colors, voxel_size)</b> · pipeline/ghost.py<br/>in ▸ 114,282 × 3 + 114,282 × 3 uint8<br/>out ▸ 17,979 × 3 — about 6.4 points per voxel<br/>DOES NOT remove the ghost. It sets the SPACING<br/>removing it costs +2.59% on the reported volume"]
            A_NORM["<b>normal_aware_filter(pts, cols, voxel, max_deviation=0.3, k=20)</b><br/>in ▸ 17,979 × 3<br/>out ▸ 17,443 × 3 — rejects 1−|dot(n, mean_n)| > 0.3 (~45°)<br/>not load-bearing: -0.08% on the volume"]
            A_MLS["<b>mls_project(pts, cols, radius_mult, polynomial=True)</b> · pipeline/ghost.py<br/>in ▸ 17,443 × 3<br/>out ▸ 17,443 × 3 moved + colours + stats dict<br/>SVD local frame, degree-2 height field, radius 4.0 × spacing<br/>THIS collapses the two ghost sheets: 1.76 mm → 0.79 mm"]
            A_PLANE --> A_DB --> A_CUBE
            A_CUBE --> A_BOXC
            A_CUBE --> A_OBJC
            A_OBJC -- "dense, uncleaned" --> A_MARK --> A_UP --> A_FIT
            A_BOXC --> A_SCALE
            A_OBJC --> A_SCALE
            A_SCALE -- "voxel 0.0050" --> A_DEDUP --> A_NORM --> A_MLS
        end

        subgraph SG3B["PHASE B · levelling"]
            B_LEVEL["<b>_level_to_ground(dense_cloud, seed)</b><br/>in ▸ cloud<br/>out ▸ R_total (3,3) — RANSAC floor normal rotated onto +Z"]
            B_ROT["<b>apply R_total</b><br/>in ▸ every sub-cloud AND the marker planes<br/>out ▸ levelled copies<br/>a plane that misses this is in a frame the meshes do not share"]
            B_LEVEL -- "R_total (3,3)" --> B_ROT
        end

        subgraph SG3C["PHASE C · close and publish"]
            C_GH["<b>_find_ground_height(dense, levelled, seed)</b><br/>in ▸ levelled cloud<br/>out ▸ floor_z = -0.6494<br/>re-fitted in the levelled frame, not reused from Phase A"]
            C_DROP["<b>_drop_below_floor(points, colours, floor_height, margin=0.008)</b><br/>in ▸ levelled cloud + floor_z<br/>out ▸ cloud above the floor, 8 mm margin"]
            C_EXT["<b>extend to floor</b> · core/fill.py<br/>in ▸ open cloud + floor_z<br/>out ▸ +1,238 wall points at 2.5 mm spacing"]
            C_CAP["<b>bottom cap</b> · core/fill.py<br/>in ▸ wall-extended cloud<br/>out ▸ closed base"]
            C_GH --> C_DROP --> C_EXT --> C_CAP
        end

        L_VOX -- "445,654 pts" --> A_PLANE
        A_MLS -- "filtered clusters" --> B_LEVEL
        A_FIT -- "marker planes" --> B_ROT
        B_ROT -- "levelled clouds" --> C_GH
    end

    F2 -- "858,554 pts" --> L_READ
    F0D -.-> A_MARK

    F3A[/"<b>03_clean/objects/leg_open.ply</b><br/>17,443 pts · levelled, filtered, closed at the floor, NOT cut<br/>the exact input a re-cut reuses"/]:::file
    F3B[/"<b>03_clean/objects/leg_no_cut.ply</b><br/>19,333 pts · the cloud the Review screen draws"/]:::file
    F3C[/"<b>03_clean/objects/box.ply</b><br/>19,573 pts · the reference, extent (0.315,0.321,0.231)"/]:::file
    F3D[/"<b>03_clean/objects/merged.ply</b><br/>35,020 pts · both objects, for inspection"/]:::file
    F3E[/"<b>03_clean/debug/cutting_line_levelled.json</b><br/>markers[]: centroid (3,), normal (3,), npts<br/>space = 'levelled' ← the ONLY file the web app may read"/]:::file
    F3F[/"<b>03_clean/debug/levelling.json</b><br/>R_total (3,3) · floor_z · note"/]:::file
    NOLEG(["<b>NO leg_cut.ply is written</b><br/>and a stale one is deleted<br/>enforced by service/jobs.py:_postcondition"]):::guard

    C_CAP --> F3A
    C_CAP --> F3B
    C_CAP --> F3C
    C_CAP --> F3D
    C_CAP --> NOLEG
    B_ROT --> F3E
    B_ROT --> F3F

%% ═══════════════ calibration branch ═══════════════
    subgraph SGCAL["STAGES 4-6 · REFERENCE CUBE ONLY — calibration, not a measurement"]
        direction TB
        CAL_R["<b>Stage 4 · alpha shape on box.ply</b><br/>in ▸ 19,573 pts<br/>out ▸ 16,106 faces, watertight, χ = 2"]
        CAL_W["<b>Stage 5 · watertight repair</b><br/>in ▸ box_recon.ply<br/>out ▸ box.ply — already closed, so unchanged"]
        CAL_V["<b>Stage 6 · volume of the cube ONLY</b><br/>in ▸ box.ply, V = 0.012164 mesh units³<br/>out ▸ linear_scale = (2744 / V)^(1/3) = 60.87 cm per unit"]
        CAL_R --> CAL_W --> CAL_V
    end

    F3C -- "19,573 pts" --> CAL_R
    FCAL[/"<b>06_volume/volumes.csv</b> — REFERENCE ROW ONLY<br/>box.ply, is_ref=True, 2744.00 cm³ by construction<br/>no subject row exists yet"/]:::file
    CAL_V --> FCAL

%% ═══════════════════════ REVIEW ═══════════════════════
    subgraph SGREV["REVIEW · the browser — web/src/components/screens/Review.tsx"]
        direction TB
        R_LOAD["<b>usePly(url)</b><br/>in ▸ leg_no_cut.ply over HTTP<br/>out ▸ THREE.BufferGeometry<br/>rotateX(-90°) sends mesh Z to scene Y"]
        R_SCALE["<b>linearScale(rows)</b> · web/src/lib/data.ts<br/>in ▸ parsed volumes.csv rows<br/>out ▸ cm per mesh unit, or null<br/>null is why the calibration branch exists"]
        R_DRAW["<b>draw the detected plane</b><br/>in ▸ cutting_line_levelled.json + scale<br/>out ▸ plane rendered on the limb, in centimetres"]
        R_DRAG["<b>person drags height / tilt / direction</b><br/>in ▸ pointer events<br/>out ▸ {centroid, normal} in LEVELLED space"]
        R_PREV["<b>splitByPlanes()</b> · CutReview.tsx<br/>in ▸ geometry + planes<br/>out ▸ instant client-side preview<br/>a PREVIEW only — never the reported number"]
        R_LOAD --> R_DRAW
        R_SCALE --> R_DRAW
        R_DRAW --> R_DRAG --> R_PREV
    end

    F3B -- "19,333 pts" --> R_LOAD
    F3E -- "detected planes" --> R_DRAW
    FCAL -- "60.87 cm/unit" --> R_SCALE

    CONFIRM{{"<b>cut confirmed?</b><br/>PERSON decides<br/>in ▸ the limb, the plane, the preview<br/>out ▸ planes[] or another adjustment"}}:::human
    R_PREV --> CONFIRM
    CONFIRM -- "adjust · POST /recut" --> R_DRAG

    FPLANES[/"<b>work/&lt;job&gt;/planes.json</b><br/>markers[]: centroid (3,), normal (3,)<br/>space = 'levelled' · at most 2"/]:::file
    CONFIRM -- "yes" --> FPLANES

%% ═══════════════════════ STAGE 3 · pass 2 ═══════════════════════
    subgraph SG3CUT["STAGE 3 --cut-only — clean.py · cut_only(stage3_dir, planes, fill_enabled)"]
        direction TB
        K_LOAD["<b>reload leg_open.ply</b><br/>in ▸ 03_clean/objects/leg_open.ply<br/>out ▸ 17,443 pts — already levelled and filtered<br/>nothing before the cut is recomputed"]
        K_RULE["<b>apply_marker_cut(cloud, planes)</b> · core/segmentation.py<br/>in ▸ cloud + 0, 1 or 2 planes<br/>out ▸ cut cloud<br/>0 ▸ no cut · 1 ▸ keep below · 2 ▸ keep between · &gt;2 refused"]
        K_CAP["<b>cap the cut face, re-close the base</b> · core/fill.py<br/>in ▸ cut cloud<br/>out ▸ 15,447 pts, closed"]
        K_LOAD --> K_RULE --> K_CAP
    end

    F3A -- "17,443 pts" --> K_LOAD
    FPLANES -- "confirmed planes" --> K_RULE
    F3G[/"<b>03_clean/objects/leg_cut.ply</b><br/>15,447 pts<br/>+ debug/cutting_line_review.json"/]:::file
    K_CAP --> F3G

%% ═══════════════════════ STAGE 4 ═══════════════════════
    subgraph SG4["STAGE 4 · surface reconstruction — reconstruct.py + workers/recons_methods_worker.py"]
        direction TB
        R_PICK["<b>_pick_method(path, method, box_method, obj_method)</b><br/>in ▸ object path + any overrides<br/>out ▸ method name — <b>poisson</b> for both by default<br/>--recon-method does NOT affect the cube; use --box-recon-method"]
        R_SUB["<b>subprocess per object</b><br/>in ▸ one PLY + method<br/>out ▸ one mesh — exiting is what frees the solver's memory"]
        R_PSR["<b>Poisson</b> · create_from_point_cloud_poisson<br/>in ▸ oriented normals, adaptive depth 6-9<br/>out ▸ mesh + per-vertex density<br/>then trim the lowest 2-5% density — REQUIRED, the<br/>extrapolated skin is what breaks the topology"]
        R_CHK["<b>_survives_repair(recon_ply)</b><br/>in ▸ the mesh just written<br/>out ▸ (ok, euler) — runs the SAME repair Stage 5 will"]
        R_OK{"<b>χ = 2?</b><br/>a single closed solid<br/>NOT merely watertight: a tunnelled surface<br/>is closed and is_watertight returns True"}
        R_KEEP["<b>keep the Poisson mesh</b><br/>closer to the points: p95 1.30 mm vs alpha's 2.39 mm"]
        R_FB["<b>rebuild THIS object with alpha_shape</b><br/>in ▸ the same cloud<br/>out ▸ mesh + re-check of χ, reported either way"]:::guard
        R_TETRA["<b>TetraMesh.create_from_point_cloud(pcd)</b><br/>in ▸ (N,3)<br/>out ▸ Delaunay complex + mapping<br/>computed ONCE and reused by every rung"]
        R_LADDER["<b>alpha ladder</b><br/>in ▸ complex + mean point spacing<br/>out ▸ one candidate mesh per rung<br/>8·10·12·14·16·20·25·30·40·55·70·90·110·140·170·200 ×"]
        R_TEST{"<b>watertight AND Euler characteristic = 2?</b><br/>in ▸ candidate mesh (trimesh, process=False)<br/>out ▸ accept / reject"}
        R_TAKE["<b>take the SMALLEST α that passes</b><br/>the tightest surface that is still one closed solid<br/>this is the GUARANTEE Poisson does not have"]
        R_RANK["<b>ranking, when no rung passed</b> (and it says so in the log)<br/>in ▸ all candidates<br/>out ▸ best by (closed, −|χ-2|, face count)"]
        R_PICK --> R_SUB --> R_PSR --> R_CHK --> R_OK
        R_OK -- "yes" --> R_KEEP
        R_OK -- "no · e.g. χ = −18 on an uncut foot" --> R_FB
        R_FB --> R_TETRA --> R_LADDER --> R_TEST
        R_TEST -- "yes" --> R_TAKE
        R_TEST -- "no rung passed" --> R_RANK
    end

    F3G -- "15,447 pts" --> R_PICK
    F3C -- "19,573 pts" --> R_PICK
    F4[/"<b>04_recon/mesh/*_recon.ply</b><br/>box_recon 8,055 v / 16,106 f · vol 0.012164<br/>leg_cut_recon 7,942 v / 15,880 f · vol 0.004796<br/>+ scene_recon.ply"/]:::file
    R_KEEP --> F4
    R_TAKE --> F4
    R_RANK --> F4

%% ═══════════════════════ STAGE 5 ═══════════════════════
    subgraph SG5["STAGE 5 · watertight repair — watertight.py + workers/meshfix_worker.py"]
        direction TB
        W_FIX["<b>_pymeshfix_repair(vertices, faces)</b><br/>in ▸ (V,3) float64 + (F,3) int32<br/>out ▸ repaired (V,3) + (F,3)<br/>fill_holes() ONLY — never drops faces, shape preserved"]
        W_O3D["<b>_o3d_fill_holes(vertices, faces)</b><br/>in ▸ still-open mesh<br/>out ▸ closed mesh — only runs if needed"]
        W_COL["<b>colour transfer</b><br/>in ▸ repaired mesh + the original coloured one<br/>out ▸ coloured mesh — cKDTree nearest neighbour, 3 retries"]
        W_VER["<b>verify: trimesh.load(process=False)</b><br/>in ▸ the written file<br/>out ▸ is_watertight, euler_number<br/>process=False is DELIBERATE: the default merge welds<br/>PyMeshFix's seam duplicates and calls an open mesh closed"]
        W_FIX --> W_O3D --> W_COL --> W_VER
    end

    F4 --> W_FIX
    F5[/"<b>05_watertight/mesh/</b><br/>box.ply / .stl · leg_cut.ply / .stl<br/>scene_colour.ply — 15,997 v, 31,986 f<br/>all watertight, χ = 2"/]:::file
    W_VER --> F5

%% ═══════════════════════ STAGE 6 ═══════════════════════
    subgraph SG6["STAGE 6 · real-world volume — pipeline/stages/volume.py (currently main's version)"]
        direction TB
        V_LOAD["<b>_load_mesh_info(path, voxel_res, auto_res)</b><br/>in ▸ mesh path<br/>out ▸ dict, or None if it will not load"]
        V_ISREF{"<b>_is_ref_mesh(path)</b><br/>in ▸ filename<br/>out ▸ bool — which mesh is the cube"}
        V_MEAS["<b>_measure_volume(mesh, ...)</b> — four tiers<br/>in ▸ trimesh<br/>out ▸ (volume, method string)<br/>1 exact signed volume ▸ the only trustworthy one<br/>2 warp flood fill · 3 trimesh voxel · 4 convex hull<br/>tiers 2-4 warn loudly — a leak once gave 0.000825 for 0.0119"]
        V_K["<b>scale from the reference</b><br/>in ▸ V_ref = 0.012164, REFERENCE_REAL = 2744 cm³<br/>out ▸ k = 225,579.0 · linear_scale = k^(1/3) = 60.87 cm/unit<br/>⚠ the cube therefore prints 2744.00 by construction"]
        V_DIM["<b>dimensions</b><br/>in ▸ AXIS-ALIGNED extents × linear_scale<br/>out ▸ cm — a tilted 14 cm cube reads 19.18 × 19.47 × 14.09<br/>because an AABB measures the diagonal"]
        V_X["<b>voxel cross-check</b> · auto_tune_voxel_res()<br/>in ▸ mesh + resolution<br/>out ▸ occupancy volume, for comparison only"]
        V_LOAD --> V_ISREF --> V_MEAS --> V_K --> V_DIM --> V_X
    end

    F5 --> V_LOAD
    F6[/"<b>06_volume/volumes.csv</b><br/>name · is_ref · volume · method · ext_x/y/z · bbox_vol<br/>real_vol_cm3 · real_vol_L · size_x/y/z_cm<br/>box 2744.00 cm³ (identity) · limb 1081.94 cm³"/]:::file
    V_X --> F6

    OUT[/"<b>leg_mesh · box_mesh · scene_mesh</b><br/>PLY (vertex colours) + STL<br/>and the measured volume"/]:::file
    F6 --> OUT

    classDef file fill:#eef4fb,stroke:#4a72a8,color:#16324f
    classDef nn fill:#ffffff,stroke:#c0392b,stroke-width:2px,color:#7b241c
    classDef human fill:#fbf3e6,stroke:#a8752a,stroke-width:2px,color:#573a10
    classDef guard fill:#eef7ee,stroke:#2a7f4f,color:#14401f
```

---

## The same thing as tables

Everything below is also in the chart. It is kept because a table is easier to
scan when you already know which function you want and only need its signature,
and easier to search than an SVG. The chart is the better way to follow the
dataflow; this is the better way to look one thing up.

Numbers throughout are one real run of `inputs/small_leg` (6 photos, RTX 4060,
re-measured 2026-08-22). They are there to give a sense of scale, not as
specifications.

---

### Stage 0 · framing gate — `pipeline/stages/prep.py`

Entry point `prepare_frames(image_folder, out_dir, band_heights=1.6, pad=0.05,
centre_on_subject=True, output_size=518, strict=True, min_frames=6, crop=True)`
→ `manifest: dict`.

| sub-process | in | out | notes |
|---|---|---|---|
| `_cube_faces(gray, dict_name)` | grayscale `(H,W) uint8` | list of 4×2 float quads | ArUco `DICT_5X5_250`. Returns marker quads, not face quads |
| `_face_corners(quad, face_cm, marker_cm)` | one marker quad | the whole cube face's 4 corners | homography from the marker's known size to the face's; recovers face area the marker only samples |
| `_cube_bbox(gray, image_pil, dict_name)` | grayscale + PIL image | `(x0,y0,x1,y1)` or `None` | union of the ArUco face quads with a GroundingDINO `cardboard box` detection. **If the detector raises, the union silently falls back to the faces alone, which is smaller — see `repo_review.md` finding 5** |
| `_leg_mask(image_pil, cube)` | PIL image + cube box | boolean mask `(H,W)` | GroundingDINO `leg` box, then SAM prompted with that box. The cube box is passed so a limb overlapping it is not selected |
| `_band_bbox(image_pil, bgr, leg_box, limb_mask)` | PIL + BGR + limb box + limb mask | `(x0,y0,x1,y1)` or `None` | GroundingDINO `cord`. **Two guards**: the box must intersect the selected limb, and it must be no larger than `BAND_MAX_LIMB_FRAC` (0.35) of the limb's mask area. Without the second, a capture with no cord gets the *leg* returned as the band — which passes the overlap test trivially |
| `trace_band_colour(bgr, box, keep_percentile=40, dilate=3)` (`core/vlm_detect.py`) | BGR image + band box | mean RGB + excess-green margin | per-column maximum-deviation trace of the cord, then **±3 rows around it**. Sampling the trace alone reports the cord's darkest pixel, which left the detector only **45** band points to fit a plane through and tilted it 27.1° from vertical against a limb leaning 19.0°. Sampling the cord's body finds **296** and tilts 19.0°. See `experiments/cut_plane_band_colour.png` |
| `_subject_bounds(cube_box, band_box, limb_mask, band_heights, pad)` | the three detections | one union box | limb is followed up to `band_heights × cube height` (1.6) so the window does not chase a whole leg out of frame |
| `_square_window(union, shape, cube, pad, ...)` | union box + frame shape | square window | full frame width, slides only vertically. Widening is impossible without leaving the photo |
| `_vggt_window(shape, target=518, patch=14)` | frame shape | the centre crop VGGT would take | used **only** as the second chance in the gate |
| `_box_fits_inside(window, box, margin_frac)` | window + a box | bool | 5% margin. Applied to the cube and to the band separately |
| `_frame_verdict(cube_seen, band_seen, cube_ok, band_ok)` | four bools | `(verdict, notes, severity)` | the whole policy, in one pure function. `verdict` is `pass`, `warning` or `reject`, ranked so the worst finding on a frame wins |
| `_debug_overlay(img, window, cube, band, ok, mode, notes, limb, verdict, ...)` | frame + all four boxes + the verdict | annotated PNG ≤1200 px | **four boxes**: magenta cube, orange limb, green band, yellow window. The limb is drawn because without it a missing band is unreadable — you cannot tell whether the detector found the wrong thing or nothing. Banner is colour-coded by verdict |
| `_read_image_bgr(path)` | file path | BGR array or `None` | `cv2.imread`, falling back to PIL + `pillow_heif`. **HEIC needs the fallback**: OpenCV cannot decode it, and a phone shoots HEIC by default |
| `_write_frame_for_vggt(image, window, should_crop, output_size, out_path)` | PIL + window | `frame_NN.png` | crops and resizes to 518², **or writes the frame raw** when it could not be framed |
| `_average_band_colour(per_frame_colours, total_frames)` | per-frame colours + how many frames were submitted | one dict, or **`None`** | median over the frames that saw a band. Returns `None` unless at least `BAND_MIN_FRAME_FRAC` (0.6) of the submitted frames agree — one detection is not corroboration, and a single false positive once set the marker colour to the floor tile |

**Written:** `00_prep/images/frame_NN.png`, `00_prep/framing.json`,
`00_prep/overlays/*.png`, `00_prep/manifest.json`.

`framing.json` keys: `submitted`, `accepted`, `all_passed`,
`band_cube_heights`, `pad_frac`, `centre_on_subject`, `marker_colour`, and per
frame `name`, `mode`, `reasons`, `severity`, `overlay`, `window`, `cube`,
`band`.

**Rejection policy**

| condition | verdict | what the frame is told |
|---|---|---|
| everything found and framed | **pass** | — |
| band missing | **warning** | marker missing — the cut must be placed by hand |
| band found but clipped | **warning** | marker out of window — the suggested cut may be off |
| cube found but clipped | **warning** | cube out of window — VGGT will centre-crop instead |
| cube not detected | **reject** | cube missing — the scale cannot be recovered |
| nothing detected | **reject** | nothing detected — no cube and no marker |
| file cannot be decoded | **reject** | file unreadable — cannot be decoded |

**A warning is a usable frame.** The distinction is what a defect *costs*. The
reference cube sets the scale of every number, so if it is not detected at all
there is nothing to recover from — that is a reject. Everything else degrades the
result without making it impossible: a clipped cube falls back to VGGT's own
centre crop, and a missing band only means the cut is placed by a person in the
review step, which it is anyway. Only rejects stop the run.

Everything not croppable is written **raw**, rejected frames included, so a
refused capture can still be inspected and `--continue-on-rejected` can proceed.

---

### Stage 1 · inference — `pipeline/stages/inference.py`

`run_inference(image_folder, device, max_frames=None, preprocess_mode="crop", ...)`
→ `(predictions: dict, seconds: float)`.

| sub-process | in | out | notes |
|---|---|---|---|
| `_load_weights()` | — | model + which checkpoint | prefers `VGGT-1B-Commercial`; falls back **loudly** to `facebook/VGGT-1B`, which is CC BY-NC-SA and not licensed for commercial use |
| `_select_frames(image_names, max_frames)` | file list | capped file list | 6 on MPS; auto elsewhere |
| `load_and_preprocess_images(mode)` | frame paths | `(S,3,518,518)` float | Stage 0 already emitted squares, so `crop` is a resize here and discards nothing. Without Stage 0 this is where 43.8% of a 9:16 photo is thrown away |
| forward pass | image tensor | nine arrays | one pass; the model is freed and the CUDA cache emptied immediately after |

**Written:** `01_inference/predictions.npz` and `01_inference/raw/`.

| array | shape | used |
|---|---|---|
| `world_points` | `(S,518,518,3)` | **yes** — the only 3D source |
| `world_points_conf` | `(S,518,518)` | yes — the Stage 2 threshold |
| `images` | `(S,3,518,518)` | yes — colours |
| `depth`, `depth_conf` | `(S,518,518,1)` | no — available to `multiview.py`, which is not wired in |
| `world_points_from_depth` | `(S,518,518,3)` | no — measured worse |
| `extrinsic`, `intrinsic`, `pose_enc` | per frame | no in the main path |

---

### Stage 2 · point cloud — `pipeline/stages/pointcloud.py`

`export_ply(predictions, output_dir, args)` → `points.ply`.

| sub-process | in | out | notes |
|---|---|---|---|
| `_extract_base_cloud(predictions, args)` | predictions + args | `(pts, cols, conf)` | picks `world_points` under `--prediction_mode pointmap` |
| confidence filter | conf array | boolean mask | `--conf_thres 45` is a **percentile**, not an absolute. 45 filters floor and object evenly |
| background masks | RGB | mask | `--mask_black_bg` / `--mask_white_bg`, both off by default |
| colour assignment | `images` | `(N,3) uint8` | each point takes its own pixel from its own view. Colours are **never averaged across views** — verified, `pointcloud.py:61` |
| spatial outlier removal | `(N,3)` | `(M,3)` | 885,470 → 858,554 |

---

### Stage 3 · segment, detect, cut — `pipeline/stages/clean.py`

Entry points `clean_and_extract(...)` (detect, optionally cut) and
`cut_only(stage3_dir, planes, fill_enabled)` (apply a confirmed cut to what the
first pass left on disk).

**Load**

| sub-process | in | out | numbers |
|---|---|---|---|
| `_load_and_thin(dense_ply)` | `points.ply` | thinned Open3D cloud | SOR `nb=20, std_ratio=2.5` 858,554 → 823,550; `voxel_down_sample(0.002)` → 445,654. The voxel size here is a **speed** setting, not a surface one |

**Phase A — original VGGT space, before levelling.** Clustering and marker
detection both degrade if run on a thinned or rotated cloud, so they run first.

| sub-process | in | out | numbers |
|---|---|---|---|
| `remove_dominant_plane` (`core/plane.py`) | thinned cloud | cloud without the floor | RANSAC removes 219,846 pts (49.3%), leaving 225,808. Without this the floor connects cube to limb and DBSCAN sees one blob |
| `detect_top_k_objects` (`core/cluster.py`) | floorless cloud | k clusters + which is the reference | DBSCAN, then cubeness `min_extent / max_extent`. box 106,832 · limb 114,282 |
| `_detect_marker_planes(cluster, axis, marker_colour)` | limb cluster + learned colour | `[{centroid, normal, npts}]` | chromaticity-space linear discriminant; restricted to the **upper 60% of the span** because the foot is wider than the calf; RANSAC plane through the selected points |
| `_ghost_filter_scales(dense_cloud)` | cloud | `(voxel_size, normal_scale)` | `0.65 × mean NN spacing` = 0.0050, shared across clusters so both are decimated identically |
| `_clean_cluster(cluster, label, voxel, scale)` | one cluster | filtered points + colours | calls the three ghost functions in order — see below |
| `_save_cluster_debug(...)` | clusters + markers | debug PLYs | inspection only |

**The ghost chain, in `pipeline/ghost.py`** — the part most often mis-attributed:

| function | in | out | what it really does |
|---|---|---|---|
| `compute_voxel_size(points, factor)` | `(N,3)` | float | `factor × mean NN distance`, `factor = GHOST_VOXEL_FACTOR = 0.65` |
| `ghost_voxel_downsample(points, colors, voxel_size)` | `(N,3)`, `(N,3)` | `(M,3)`, `(M,3)` | 114,282 → 17,979, about 6.4 points per voxel. **Does not remove the ghost** — two sheets 2 mm apart fall in different voxels. It sets the *spacing*, which is what gives MLS a wide enough radius. Removing it costs **+2.59%** on the reported volume |
| `normal_aware_filter(points, colors, voxel, max_deviation=0.3, k=20)` | filtered cloud | smaller cloud | 17,979 → 17,443 (-3.2%). Rejects `1 − |dot(n, mean_n)| > 0.3`, about 45°. Not load-bearing on the volume |
| `mls_project(points, colors, radius_mult, min_neighbors=8, polynomial=True)` | `(N,3)` | `(N,3)`, colours, stats | **this is what collapses the ghost.** SVD local frame, degree-2 height field, radius 4.0 × spacing. Moves points rather than deleting them; shell 1.76 mm → 0.79 mm. Returns `{spacing, radius, median_move, p95_move, skipped}` |

**Phase B — levelling**

| sub-process | in | out | notes |
|---|---|---|---|
| `_level_to_ground(dense_cloud, seed)` | cloud | `R_total (3,3)` | RANSAC floor normal rotated onto +Z |
| rotate everything | all sub-clouds **and the marker planes** | levelled copies | the marker planes must take the same rotation or the cut is applied in a frame the meshes do not share. This is the bug `cutting_line.json` vs `cutting_line_levelled.json` exists to prevent |

**Phase C — close and publish**

| sub-process | in | out | numbers |
|---|---|---|---|
| `_find_ground_height(...)` | levelled cloud | `floor_z` | re-fitted in the levelled frame, not reused from Phase A |
| `_drop_below_floor(points, colours, floor_height, margin=0.008)` | levelled cloud | cloud above the floor | 8 mm margin |
| extend to floor + bottom cap (`core/fill.py`) | open cloud | closed cloud | +1,238 wall points at 2.5 mm spacing |
| `apply_marker_cut` (`core/segmentation.py`) | cloud + planes | cut cloud | 0 planes no cut · 1 keeps below · 2 keeps between · caps at two |

**Written by the `--no-cut` pass:** `objects/leg_open.ply` (17,443 — levelled,
filtered, closed at the floor, *not* cut: the input a re-cut reuses),
`objects/leg_no_cut.ply` (19,333 — the review cloud), `objects/box.ply`
(19,573), `objects/merged.ply`, `debug/cutting_line_levelled.json`,
`debug/levelling.json`. **No `leg_cut.ply`**, and a stale one is deleted.

**Written by `cut_only`:** `objects/leg_cut.ply` (15,447) and
`debug/cutting_line_review.json`.

---

### Stage 4 · surface reconstruction — `pipeline/stages/reconstruct.py`

`reconstruct_mesh_stage(object_paths, output_dir, seed, method, ...)`. Each
object is reconstructed in its own subprocess (`workers/recons_methods_worker.py`),
which is what releases the memory Open3D's tetrahedralisation holds.

| sub-process | in | out | notes |
|---|---|---|---|
| `_pick_method(path, method, box_method, obj_method)` | path + overrides | method name | **`poisson` for both by default.** `--box-recon-method` / `--obj-recon-method` override per object. Note `--recon-method` does **not** reach the cube — it is resolved from `_DEFAULT_BOX_METHOD` |
| `_survives_repair(recon_ply)` | the mesh just written | `(ok, euler)` | runs the same repair Stage 5 will and asks whether the result is χ = 2. **Not** whether it is watertight: a tunnelled surface is closed and `is_watertight` returns True. On a check error it returns `ok=True` and says so, rather than swapping method because trimesh hiccupped |
| alpha fallback | the same cloud | alpha-shape mesh | fires per object when `_survives_repair` says no, then re-checks and reports either way. Poisson gives χ = −18 on `short_leg`'s uncut foot; alpha rebuilds it at χ = 2 |
| `TetraMesh.create_from_point_cloud` | `(N,3)` | Delaunay complex | computed **once** and reused for every rung — the expensive part |
| alpha ladder | complex | candidate meshes | 8, 10, 12, 14, 16, 20, 25, 30, 40, 55, 70, 90, 110, 140, 170, 200 × point spacing. The ladder used to stop at 90× and a cropped limb needs up to 140× |
| selection test | candidate | accept / reject | **watertight AND Euler characteristic 2.** Chosen: box 55×, limb 25× |
| fallback ranking | all candidates | best effort | closed first, then `|χ − 2|`, then face count — and it says so in the log |

The α ladder is now the **fallback path** rather than the default, but the rule
it implements is unchanged and is the reason it is the fallback: it is the only
method here that *guarantees* a single closed solid, because it selects on that
property. Why the choice is measured this way is in
[`experiments/recon_method_comparison.png`](experiments/recon_method_comparison.png):
Poisson and ball pivoting fit the points **better** and still cannot be
measured, because after repair they close at χ = 22 and χ = 256 instead of 2.

**Written:** `04_recon/mesh/*_recon.ply` — box 16,106 faces, limb 15,688.

---

### Stage 5 · watertight repair — `pipeline/stages/watertight.py`

Runs `workers/meshfix_worker.py` as a subprocess.

| sub-process | in | out | notes |
|---|---|---|---|
| `_pymeshfix_repair(vertices, faces)` | mesh | mesh | `fill_holes()` only — never drops faces, so the shape and the retention are preserved |
| `_o3d_fill_holes(vertices, faces)` | mesh | mesh | only if still open |
| colour transfer | repaired mesh + original | coloured mesh | cKDTree nearest neighbour, 3 retries for transient C++ crashes |
| verification | mesh file | bool | `trimesh.load(process=False)` **deliberately** — the default merge welds PyMeshFix's intentional seam duplicates and reports an open mesh as watertight |

**Written:** `05_watertight/mesh/*.ply` + `.stl`, and `scene_colour.ply`.

---

### Stage 6 · real-world volume — `pipeline/stages/volume.py`

> Currently **main's version**, reverted pending review by the stage's author.
> The alternative method is parked as a commented block at the bottom of the same
> file. See [`stage06_experiments.md`](stage06_experiments.md).

| sub-process | in | out | notes |
|---|---|---|---|
| `_load_mesh_info(path, voxel_res, auto_res)` | mesh path | dict or `None` | |
| `_is_ref_mesh(path)` | path | bool | filename rule — which mesh is the cube |
| `_measure_volume(mesh, ...)` | mesh | `(volume, method)` | four tiers: **1** exact signed volume · 2 warp flood fill · 3 trimesh voxel · 4 convex hull. Tiers 2-4 warn loudly. A non-watertight mesh once returned 0.000825 instead of 0.0119 because the flood fill escaped through a hole |
| `auto_tune_voxel_res(...)` | mesh | resolution | for the cross-check only |
| scale | reference volume | `linear_scale` | `k = 2744 / V_ref`; `linear_scale = k^(1/3)` = 60.87 cm/unit. **The cube therefore reports exactly 2744.00 cm³ on every run — an identity, not a measurement** |
| dimensions | mesh | cm | axis-aligned extents × `linear_scale`, so a tilted 14 cm cube reads 19.18 × 19.47 × 14.09 — an AABB measures the diagonal |

**Written:** `06_volume/volumes.csv` — columns `name, is_ref, volume, method,
ext_x, ext_y, ext_z, bbox_vol, real_vol_cm3, real_vol_L, size_x_cm, size_y_cm,
size_z_cm`.

---

## What the chart does not show

Stated so it is not mistaken for complete.

- **Error paths.** Every stage has failure branches; drawing them would double
  the chart without adding to what it is for. The one exception is the deferred-cut
  invariant, which is drawn because violating it is the failure that matters most.
- **The three latent silent failures** in `repo_review.md` items 3, 4 and 5. They
  are `except` blocks inside boxes drawn here as single steps.
- **`pipeline/multiview.py`.** Written, documented, never wired into Stage 2.
- **The web app's own screens** beyond Review. Those are in
  [`web_explaination.md`](web_explaination.md) and figures 7-8 of
  [`pipeline_flowchart.md`](pipeline_flowchart.md).
