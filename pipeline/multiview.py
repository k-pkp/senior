"""Multi-view consistency filtering.

VGGT emits duplicate copies of a surface offset a few millimetres from the true
one ("ghost" sheets). Neither of the geometric filters can remove them:

  - voxel dedup only merges points sharing a cell, and two sheets 2 mm apart
    usually fall in different cells;
  - the normal-aware filter rejects points whose normal disagrees with their
    neighbourhood, but a ghost sheet is *parallel* to the true surface, so its
    normals agree perfectly. It is blind to exactly this case.

What does distinguish a ghost is that it sits at the wrong depth. Reproject a
point into the other cameras and compare the geometric depth (from the camera
pose) against the depth head's own prediction at that pixel. A true surface
point agrees in many views; a ghost only agrees in the view that produced it.

This needs `depth`, `extrinsic` and `intrinsic`, which exist only as Stage 1
outputs — so it belongs in Stage 2, as a per-pixel mask alongside the
confidence filter.
"""
import numpy as np


def multiview_agreement(world_points, depth, extrinsic, intrinsic,
                        rel_threshold=0.05):
    """Count, per point, how many other views agree on its depth.

    Args:
        world_points: (S, H, W, 3)
        depth:        (S, H, W) or (S, H, W, 1)
        extrinsic:    (S, 3, 4) cam-from-world, OpenCV convention
        intrinsic:    (S, 3, 3)
        rel_threshold: relative depth difference counted as agreement.

    Returns:
        (S, H, W) int32 count of agreeing *other* views (0..S-1).
    """
    if depth.ndim == 4:
        depth = depth[..., 0]
    S, H, W = world_points.shape[:3]
    agree = np.zeros((S, H, W), dtype=np.int32)
    if S <= 1:
        return agree

    for i in range(S):
        pts = world_points[i].reshape(-1, 3).astype(np.float64)
        homo = np.hstack([pts, np.ones((len(pts), 1))])
        count = np.zeros(len(pts), dtype=np.int32)

        for j in range(S):
            if i == j:
                continue
            cam = homo @ extrinsic[j].T
            dz = cam[:, 2]
            fx, fy = float(intrinsic[j, 0, 0]), float(intrinsic[j, 1, 1])
            cx, cy = float(intrinsic[j, 0, 2]), float(intrinsic[j, 1, 2])

            ok = dz > 1e-6
            safe = np.where(ok, dz, np.inf)
            u = np.rint(fx * cam[:, 0] / safe + cx)
            v = np.rint(fy * cam[:, 1] / safe + cy)
            u = np.nan_to_num(u, nan=-1, posinf=-1, neginf=-1).astype(np.int32)
            v = np.nan_to_num(v, nan=-1, posinf=-1, neginf=-1).astype(np.int32)
            ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not ok.any():
                continue

            pred = np.full(len(pts), np.inf)
            pred[ok] = depth[j, v[ok], u[ok]]
            rel = np.abs(pred - dz) / (np.abs(dz) + 1e-8)
            count += ((rel < rel_threshold) & ok).astype(np.int32)

        agree[i] = count.reshape(H, W)

    return agree


def multiview_mask(world_points, depth, extrinsic, intrinsic,
                   min_views=2, rel_threshold=0.05, verbose=True):
    """Boolean (S, H, W) mask keeping points corroborated by >= min_views.

    min_views is a real trade-off, not a free win: genuinely occluded surface is
    visible in only one or two cameras and will be discarded along with the
    ghosts. With 6 frames, 2 is permissive and 3 is strict.
    """
    agree = multiview_agreement(world_points, depth, extrinsic, intrinsic,
                                rel_threshold)
    mask = agree >= min_views
    if verbose:
        total = mask.size
        print(f"  Multi-view consistency: {mask.sum():,}/{total:,} points kept "
              f"({mask.mean()*100:.1f}%) at >={min_views} of "
              f"{world_points.shape[0]-1} other views, rel<{rel_threshold}")
    return mask
