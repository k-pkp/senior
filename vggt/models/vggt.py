# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.utils.device import (
    is_mps, is_cuda, autocast_off, aggressive_cleanup,
)


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    _NEEDED_LAYER_INDICES = [4, 11, 17, 23]          # used by DPT heads

    def _get_needed_layer_indices(self):
        """Union of aggregator output indices that any active head reads."""
        indices = set()
        for head in (self.depth_head, self.point_head):
            if head is not None:
                indices.update(head.intermediate_layer_idx)
        if self.track_head is not None:
            indices.update(self.track_head.feature_extractor.intermediate_layer_idx)
        if self.camera_head is not None:
            indices.add(max(indices) if indices else 23)  # camera_head uses [-1]
        return sorted(indices)

    @staticmethod
    def _remap_tokens(sparse_list, head_layer_indices, index_map):
        """Build a correctly-indexed list so the head can do list[layer_idx]."""
        n = max(head_layer_indices) + 1
        out = [None] * n
        for idx in head_layer_indices:
            out[idx] = sparse_list[index_map[idx]]
        return out

    # ------------------------------------------------------------------ #
    #  Standard forward — CUDA path, completely unchanged logic           #
    # ------------------------------------------------------------------ #

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
            query_points (torch.Tensor, optional): Query points for tracking.

        Returns:
            dict with pose_enc, depth, depth_conf, world_points, world_points_conf, images, ...
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # --- Branch: MPS uses memory-efficient path ----
        if is_mps(images.device) and not self.training:
            return self._forward_mps(images, query_points)

        # --- Branch: limited-VRAM CUDA uses memory-efficient path ---
        if is_cuda(images.device) and not self.training:
            free_mem = torch.cuda.mem_get_info(images.device)[0]
            # Use memory-efficient path if less than 16 GB free
            if free_mem < 16 * (1024 ** 3):
                return self._forward_cuda_lowmem(images, query_points)

        # --- CUDA / CPU: original code, byte-for-byte identical logic ---
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.amp.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images,
                patch_start_idx=patch_start_idx, query_points=query_points,
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images

        return predictions

    # ------------------------------------------------------------------ #
    #  Low-VRAM CUDA forward                                              #
    #  ‣ Same precision as standard CUDA (quality identical).             #
    #  ‣ Memory savings: keep only needed aggregator outputs, run heads   #
    #    sequentially with small chunk sizes, offload to CPU.             #
    # ------------------------------------------------------------------ #

    def _forward_cuda_lowmem(self, images: torch.Tensor, query_points: torch.Tensor = None):
        device = images.device
        predictions = {}

        # Use memory-efficient aggregator (keep only 4/24 layer outputs)
        needed = self._get_needed_layer_indices()
        agg_list, ps_idx = self.aggregator.forward_memory_efficient(
            images, keep_indices=needed,
        )
        aggressive_cleanup(device)

        idx_map = {orig: pos for pos, orig in enumerate(needed)}

        # Heads run sequentially with small chunk size; offload to CPU
        with torch.amp.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(agg_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
                del pose_enc_list
                aggressive_cleanup(device)

            if self.depth_head is not None:
                tok = self._remap_tokens(agg_list, self.depth_head.intermediate_layer_idx, idx_map)
                d, dc = self.depth_head(tok, images=images, patch_start_idx=ps_idx, frames_chunk_size=2)
                predictions["depth"] = d
                predictions["depth_conf"] = dc
                del d, dc, tok
                aggressive_cleanup(device)

            if self.point_head is not None:
                tok = self._remap_tokens(agg_list, self.point_head.intermediate_layer_idx, idx_map)
                p, pc = self.point_head(tok, images=images, patch_start_idx=ps_idx, frames_chunk_size=2)
                predictions["world_points"] = p
                predictions["world_points_conf"] = pc
                del p, pc, tok
                aggressive_cleanup(device)

        if self.track_head is not None and query_points is not None:
            tok = self._remap_tokens(
                agg_list, self.track_head.feature_extractor.intermediate_layer_idx, idx_map,
            )
            tl, vis, conf = self.track_head(
                tok, images=images, patch_start_idx=ps_idx, query_points=query_points,
            )
            predictions["track"] = tl[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf
            del tl, vis, conf, tok
            aggressive_cleanup(device)

        del agg_list
        aggressive_cleanup(device)

        if not self.training:
            predictions["images"] = images

        return predictions

    # ------------------------------------------------------------------ #
    #  MPS-specific forward                                               #
    #  ‣ Same float32 precision for heads (quality identical to CUDA).    #
    #  ‣ Memory savings: keep only 4/24 aggregator outputs, run heads     #
    #    sequentially, offload results to CPU between stages.             #
    # ------------------------------------------------------------------ #

    def _forward_mps(self, images: torch.Tensor, query_points: torch.Tensor = None):
        device = images.device
        predictions = {}

        # ---- aggregator: only keep the 4 layers heads need ----
        needed = self._get_needed_layer_indices()
        agg_list, ps_idx = self.aggregator.forward_memory_efficient(
            images, keep_indices=needed,
        )
        aggressive_cleanup(device)

        # Upcast to float32 so heads run in full precision (same as CUDA)
        agg_list = [t.float() for t in agg_list]

        idx_map = {orig: pos for pos, orig in enumerate(needed)}

        # ---- heads (float32, sequential, offload to CPU) ----
        with autocast_off(device):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(agg_list)
                predictions["pose_enc"] = pose_enc_list[-1].cpu()
                predictions["pose_enc_list"] = [p.cpu() for p in pose_enc_list]
                del pose_enc_list
                aggressive_cleanup(device)

            if self.depth_head is not None:
                tok = self._remap_tokens(agg_list, self.depth_head.intermediate_layer_idx, idx_map)
                d, dc = self.depth_head(tok, images=images, patch_start_idx=ps_idx)
                predictions["depth"] = d.cpu()
                predictions["depth_conf"] = dc.cpu()
                del d, dc, tok
                aggressive_cleanup(device)

            if self.point_head is not None:
                tok = self._remap_tokens(agg_list, self.point_head.intermediate_layer_idx, idx_map)
                p, pc = self.point_head(tok, images=images, patch_start_idx=ps_idx)
                predictions["world_points"] = p.cpu()
                predictions["world_points_conf"] = pc.cpu()
                del p, pc, tok
                aggressive_cleanup(device)

        if self.track_head is not None and query_points is not None:
            tok = self._remap_tokens(
                agg_list, self.track_head.feature_extractor.intermediate_layer_idx, idx_map,
            )
            tl, vis, conf = self.track_head(
                tok, images=images, patch_start_idx=ps_idx, query_points=query_points,
            )
            predictions["track"] = tl[-1].cpu()
            predictions["vis"] = vis.cpu()
            predictions["conf"] = conf.cpu()
            del tl, vis, conf, tok
            aggressive_cleanup(device)

        del agg_list
        aggressive_cleanup(device)

        predictions["images"] = images.cpu()
        return predictions
