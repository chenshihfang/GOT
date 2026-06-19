# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from cotracker.models.core.model_utils import sample_features4d, sample_features5d
from cotracker.models.core.embeddings import (
    get_2d_embedding,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)

from cotracker.models.core.cotracker.blocks import (
    Mlp,
    BasicEncoder,
    AttnBlock,
    CorrBlock,
    Attention,
)

# torch.manual_seed(50)

wLST = True
# wLST = False
print("wLST", wLST)

eva_time = False
# eva_time = True

if eva_time:

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start3 = torch.cuda.Event(enable_timing=True)
    end3 = torch.cuda.Event(enable_timing=True)
    start4 = torch.cuda.Event(enable_timing=True)
    end4 = torch.cuda.Event(enable_timing=True)
    # start.record()
    # end.record()
    # # Waits for everything to finish running
    # torch.cuda.synchronize()
    # print("XXX", start.elapsed_time(end))


class CoTracker2(nn.Module):
    def __init__(
        self,
        window_len=8,
        stride=4,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
    ):
        super(CoTracker2, self).__init__()
        self.window_len = window_len
        self.stride = stride
        self.hidden_dim = 256
        self.latent_dim = 128
        self.add_space_attn = add_space_attn
        self.fnet = BasicEncoder(output_dim=self.latent_dim)
        self.num_virtual_tracks = num_virtual_tracks
        self.model_resolution = model_resolution
        self.input_dim = 456
        self.updateformer = EfficientUpdateFormer(
            space_depth=6,
            time_depth=6,
            input_dim=self.input_dim,
            hidden_size=384,
            output_dim=self.latent_dim + 2,
            mlp_ratio=4.0,
            add_space_attn=add_space_attn,
            num_virtual_tracks=num_virtual_tracks,
        )

        time_grid = torch.linspace(0, window_len - 1, window_len).reshape(1, window_len, 1)

        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(self.input_dim, time_grid[0])
        )

        self.register_buffer(
            "pos_emb",
            get_2d_sincos_pos_embed(
                embed_dim=self.input_dim,
                grid_size=(
                    model_resolution[0] // stride,
                    model_resolution[1] // stride,
                ),
            ),
        )
        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.track_feat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.vis_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
        )

    def forward_window(
        self,
        fmaps,
        coords,
        SideNetwork_D,
        SideNetwork_U,
        track_feat=None,
        vis=None,
        track_mask=None,
        attention_mask=None,
        iters=4,
    ):
        # B = batch size
        # S = number of frames in the window)
        # N = number of tracks
        # C = channels of a point feature vector
        # E = positional embedding size
        # LRR = local receptive field radius
        # D = dimension of the transformer input tokens

        # track_feat = B S N C
        # vis = B S N 1
        # track_mask = B S N 1
        # attention_mask = B S N

        B, S_init, N, __ = track_mask.shape
        B, S, *_ = fmaps.shape

        track_mask = F.pad(track_mask, (0, 0, 0, 0, 0, S - S_init), "constant")
        track_mask_vis = (
            torch.cat([track_mask, vis], dim=-1).permute(0, 2, 1, 3).reshape(B * N, S, 2)
        )

        corr_block = CorrBlock(
            fmaps,
            num_levels=4,
            radius=3,
            padding_mode="border",
        )

        sampled_pos_emb = (
            sample_features4d(self.pos_emb.repeat(B, 1, 1, 1), coords[:, 0])
            .reshape(B * N, self.input_dim)
            .unsqueeze(1)
        )  # B E N -> (B N) 1 E

        if eva_time:
            print("iters s")
            start2.record()

        # print("fmaps.shape", fmaps.shape) # torch.Size([1, 8, 128, 96, 128])

        # for i, fmaps_ in enumerate(fmaps):
        #     print("fmaps_.shape:", fmaps_.shape) # torch.Size([8, 128, 96, 128])
        # input()

        latest_track_feat_D = 0  # Initialize outside the loop

        coord_preds = []
        for __ in range(iters):
            coords = coords.detach()  # B S N 2

            if eva_time:
                print("corr s")
                start4.record()

            corr_block.corr(track_feat)

            # Sample correlation features around each point
            fcorrs = corr_block.sample(coords)  # (B N) S LRR
            # print("fcorrs.shape", fcorrs.shape) # torch.Size([128+sup, 8, 196])

            # Get the flow embeddings
            flows = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 2)
            flow_emb = get_2d_embedding(flows, 64, cat_coords=True)  # N S E

            track_feat_ = track_feat.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            transformer_input = torch.cat([flow_emb, fcorrs, track_feat_, track_mask_vis], dim=2)
            x = transformer_input + sampled_pos_emb + self.time_emb
            x = x.view(B, N, S, -1)  # (B N) S D -> B N S D

            if eva_time:
                end4.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                print("corr", start4.elapsed_time(end4))

            if eva_time:
                print("updateformer s")
                start4.record()

            delta = self.updateformer(
                x,
                attention_mask.reshape(B * S, N),  # B S N -> (B S) N
            )


            # print("x.shape", x.shape)  # torch.Size([1, 128+sup, 8, 456])
            # print("delta.shape", delta.shape)  # torch.Size([1, 128+sup, 8, 130])

            delta_coords = delta[..., :2].permute(0, 2, 1, 3)
            coords = coords + delta_coords
            coord_preds.append(coords * self.stride)  # 

            # print("delta_coords.shape", delta_coords.shape)  # shape torch.Size([1, 8, 128+sup, 2])
            # print("coords.shape", coords.shape)  #  torch.Size([1, 8, 128+sup, 2])

            if eva_time:
                end4.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                print("updateformer", start4.elapsed_time(end4))

            if eva_time:
                print("track_feat_updater s")
                start4.record()

            delta_feats_ = delta[..., 2:].reshape(B * N * S, self.latent_dim)
            track_feat_ = track_feat.permute(0, 2, 1, 3).reshape(B * N * S, self.latent_dim)
            track_feat_ = self.track_feat_updater(self.norm(delta_feats_)) + track_feat_
            track_feat = track_feat_.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # (B N S) C -> B S N C # B 8 4 128

            if eva_time:
                end4.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                print("track_feat_updater", start4.elapsed_time(end4))

            ### PT_ToMP
            # print("track_feat_ori.shape", track_feat.shape, track_feat.requires_grad) # torch.Size([1, 8, 164, 128]) False
            with torch.enable_grad():    
                if wLST:
                    # print("track_feat.shape", track_feat.shape, track_feat.requires_grad) # False
                    # with torch.cuda.amp.autocast(enabled=False):
                    track_feat_D = SideNetwork_D(track_feat) # True
                    # print("track_feat_D.shape", track_feat_D.shape, track_feat_D.requires_grad) # True
                    latest_track_feat_D += track_feat_D  # Accumulate the output
                    # print("latest_track_feat_D.shape", latest_track_feat_D.shape, latest_track_feat_D.requires_grad) # True torch.Size([1, 8, 128, 64])
                ### 

        if eva_time:
            end2.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            print("iters", start2.elapsed_time(end2))

        # print("track_feat 1", track_feat.shape, track_feat.device, track_feat.requires_grad, track_feat.sum()) # torch.Size([1, 8, 164, 128]) False

        # track_feat = track_feat.detach() 
        # print("track_feat 2", track_feat.shape, track_feat.device, track_feat.requires_grad, track_feat.sum())
        
        with torch.enable_grad():  
            if wLST:      
                # with torch.cuda.amp.autocast(enabled=False):
                track_feat_U = SideNetwork_U(latest_track_feat_D)
                # print("track_feat_U.shape", track_feat_U.shape)
            #     print("track_feat 3", track_feat.shape, track_feat.device, track_feat.requires_grad, track_feat.sum())

        # print("track_feat_U.shape", track_feat_U.shape, track_feat_U.requires_grad) # torch.Size([1, 8, 128, 128]) True
        # print("latest_track_feat_D.shape 2", latest_track_feat_D.shape, latest_track_feat_D.requires_grad) # True
        # print("track_feat.shape", track_feat.shape, track_feat.requires_grad) # False w sup

        if wLST:  
            track_feat = track_feat_U + track_feat.detach()
            
        # input()

        vis_pred = self.vis_predictor(track_feat).reshape(B, S, N)

        # print("vis_pred ", vis_pred.shape, vis_pred.device, vis_pred.requires_grad) # False w sup
        # input()
        return coord_preds, vis_pred
        # len(coord_preds) 6
        # vis_pred.shape torch.Size([2, 8, 4])

    def get_track_feat(self, fmaps, queried_frames, queried_coords):
        sample_frames = queried_frames[:, None, :, None]
        sample_coords = torch.cat(
            [
                sample_frames,
                queried_coords[:, None],
            ],
            dim=-1,
        )
        sample_track_feats = sample_features5d(fmaps, sample_coords)
        return sample_track_feats

    def init_video_online_processing(self):
        self.online_ind = 0
        self.online_track_feat = None
        self.online_coords_predicted = None
        self.online_vis_predicted = None

    ### ToMP label part
    def scale_and_permute_labels(self, train_label, target_height, target_width):
        # Scale the label to match the spatial dimensions of fmaps
        # with torch.cuda.amp.autocast(enabled=False):
        train_label_scaled = F.interpolate(train_label.float(), size=(target_height, target_width), mode='bilinear', align_corners=False)
        # print("Scaled train_label shape:", train_label_scaled.shape) # torch.Size([1, 3, 96, 128])
        
        # Permute the scaled label to match the dimension order of fmaps
        train_label_permuted = train_label_scaled.permute(1, 0, 2, 3)  # [B, 2, H, W]
        # print("Permuted train_label shape:", train_label_permuted.shape) # torch.Size([3, 1, 96, 128])
        
        return train_label_permuted

    def expand_labels_to_fmaps(self, train_label_permuted, fmaps):
        # Expand the permuted labels to match the channel dimensionality of fmaps for addition
        train_label_expanded = train_label_permuted.unsqueeze(2).expand(-1, -1, fmaps.size(2), -1, -1)
        # print("Expanded train_label shape:", train_label_expanded.shape) # torch.Size([3, 1, 128, 96, 128])
        
        return train_label_expanded


    def process_labels(self, fmaps, train_label_first, train_label_last, target_height=96, target_width=128):

        train_label_first_permuted = self.scale_and_permute_labels(train_label_first, target_height, target_width)
        if train_label_last is not None:
            train_label_last_permuted = self.scale_and_permute_labels(train_label_last, target_height, target_width)
        
        train_label_first_expanded = self.expand_labels_to_fmaps(train_label_first_permuted, fmaps)
        if train_label_last is not None:
            train_label_last_expanded = self.expand_labels_to_fmaps(train_label_last_permuted, fmaps)
        
        # print("Output shapes from process_labels:")
        # print("train_label_first_expanded shape:", train_label_first_expanded.shape) # torch.Size([2, 1, 128, 96, 128])

        if train_label_last is None:
            train_label_last_expanded = None      
            # print("train_label_last_expanded:", train_label_last_expanded)
        # else:
            # print("train_label_last_expanded shape:", train_label_last_expanded.shape) # torch.Size([2, 1, 128, 96, 128])

        return train_label_first_expanded, train_label_last_expanded


    def add_labels_to_fmaps(self, fmaps, train_label_first_expanded, \
                                        train_label_last_expanded):
    # , train_ltrb_target_second_emb):
        # Print shapes before addition
        # print("train_ltrb_target_second_emb:", train_ltrb_target_second_emb.shape)  # torch.Size([3, 128, 96, 128])
        # print("Shape of fmaps before addition:", fmaps.shape)  # torch.Size([3, 8, 128, 96, 128])
        # print("Shape of train_label_first_expanded:", train_label_first_expanded.shape)  # torch.Size([3, 1, 128, 96, 128])
        # if train_label_last_expanded is not None:
        #     print("Shape of train_label_last_expanded:", train_label_last_expanded.shape)  # torch.Size([3, 1, 128, 96, 128])

        ##### if not pre proceess train_label to emb
        # Expand the labels to match the number of frames in fmaps
        # train_label_first_expanded = train_label_first_expanded.expand(-1, fmaps.size(1), -1, -1, -1)
        # print("Expanded train_label_first shape:", train_label_first_expanded.shape)  # torch.Size([3, 32, 128, 96, 128])
        
        # if train_label_last_expanded is not None:
        #     train_label_last_expanded = train_label_last_expanded.expand(-1, fmaps.size(1), -1, -1, -1)
            # print("Expanded train_label_last shape:", train_label_last_expanded.shape)  # torch.Size([3, 32, 128, 96, 128])
        ##### if not pre proceess train_label to emb

        # print("train_label_first_expanded.shape", train_label_first_expanded.shape) #  torch.Size([3, 1, 128, 96, 128]) or torch.Size([3, 6, 128, 96, 128])
        # print("train_label_last_expanded.shape", train_label_last_expanded.shape) #  torch.Size([3, 1, 128, 96, 128])

        # print("fmaps.shape", fmaps.shape) #  torch.Size([3, 8, 128, 96, 128])
        # print("fmaps.shape[1]", fmaps.shape[1]) # 8
        # print("(fmaps.shape[1]) // 2", (fmaps.shape[1]) // 2) #4

        # Add train_label_first to the first frame and train_label_last to the last frame of each sequence in fmaps
        fmaps[:, 0, :, :, :] += train_label_first_expanded[:, 0, :, :, :]

        # fmaps[:, :6, :, :, :] += train_label_first_expanded[:, :6, :, :, :]

        if train_label_last_expanded is not None:
            fmaps[:, (fmaps.shape[1]) // 2, :, :, :] += train_label_last_expanded[:, 0, :, :, :]

        # if train_label_last_expanded is not None:
        #     fmaps[:, -1, :, :, :] += train_label_last_expanded[:, 0, :, :, :]

            # Print shape after addition
            # print("Shape of fmaps after addition:", fmaps.shape) # torch.Size([3, 32, 128, 96, 128])

            # print("train_ltrb_target_second_emb:", train_ltrb_target_second_emb.shape)  # torch.Size([3, 128, 96, 128])
            # Expand dimensions of train_ltrb_target_second_emb to match fmaps
            # train_ltrb_target_second_emb = train_ltrb_target_second_emb.unsqueeze(1)  # Add a dimension at position 1
            # print("train_ltrb_target_second_emb:", train_ltrb_target_second_emb.shape)  #  torch.Size([3, 1, 128, 96, 128])

            # Add train_ltrb_target_second_emb to the last element of the second dimension of fmaps
            # fmaps[:, -1, :, :, :] += train_ltrb_target_second_emb[:, -1, :, :, :]

            # Print the shape of fmaps after addition
            # print("Shape of fmaps after addition:", fmaps.shape)  # Should be torch.Size([3, 32, 128, 96, 128])


        return fmaps
    ### ToMP label part

    def forward(self, 
                SideNetwork_D, 
                SideNetwork_U,
                # train_ltrb_target_second_emb, 
                train_label_first_batch, 
                train_label_last_batch, # mid
                video, 
                queries, iters=4, is_train=False, is_online=False):
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3]): input videos.
            queries (FloatTensor[B, N, 3]): point queries.
            iters (int, optional): number of updates. Defaults to 4.
            is_train (bool, optional): enables training mode. Defaults to False.
            is_online (bool, optional): enables online mode. Defaults to False. Before enabling, call model.init_video_online_processing().

        Returns:
            - coords_predicted (FloatTensor[B, T, N, 2]):
            - vis_predicted (FloatTensor[B, T, N]):
            - train_data: `None` if `is_train` is false, otherwise:
                - all_vis_predictions (List[FloatTensor[B, S, N, 1]]):
                - all_coords_predictions (List[FloatTensor[B, S, N, 2]]):
                - mask (BoolTensor[B, T, N]):
        """
        B, T, C, H, W = video.shape
        B, N, __ = queries.shape
        S = self.window_len
        device = queries.device

        # B = batch size
        # S = number of frames in the window of the padded video
        # S_trimmed = actual number of frames in the window
        # N = number of tracks
        # C = color channels (3 for RGB)
        # E = positional embedding size
        # LRR = local receptive field radius
        # D = dimension of the transformer input tokens

        # video = B T C H W
        # queries = B N 3
        # coords_init = B S N 2
        # vis_init = B S N 1

        assert S >= 2  # A tracker needs at least two frames to track something
        if is_online:
            assert T <= S, "Online mode: video chunk must be <= window size."
            assert self.online_ind is not None, "Call model.init_video_online_processing() first."
            assert not is_train, "Training not supported in online mode."
        step = S // 2  # How much the sliding window moves at every step
        video = 2 * (video / 255.0) - 1.0

        # print("video.shape", video.shape) #torch.Size([B, T, 3, 384, 512])

        # The first channel is the frame number
        # The rest are the coordinates of points we want to track
        queried_frames = queries[:, :, 0].long()

        queried_coords = queries[..., 1:]
        queried_coords = queried_coords / self.stride

        # We store our predictions here
        coords_predicted = torch.zeros((B, T, N, 2), device=device)
        vis_predicted = torch.zeros((B, T, N), device=device)
        if is_online:
            if self.online_coords_predicted is None:
                # Init online predictions with zeros
                self.online_coords_predicted = coords_predicted
                self.online_vis_predicted = vis_predicted
            else:
                # Pad online predictions with zeros for the current window
                pad = min(step, T - step)
                coords_predicted = F.pad(
                    self.online_coords_predicted, (0, 0, 0, 0, 0, pad), "constant"
                )
                vis_predicted = F.pad(self.online_vis_predicted, (0, 0, 0, pad), "constant")
        all_coords_predictions, all_vis_predictions = [], []

        # Pad the video so that an integer number of sliding windows fit into it
        # TODO: we may drop this requirement because the transformer should not care
        # TODO: pad the features instead of the video
        # pad frames are added to the end of the video sequence, 
        # and these frames are replicas of the last frame of the original video 
        # (because "replicate" as the padding mode).
        # pad 2 T 30
        pad = S - T if is_online else (S - T % S) % S  # We don't want to pad if T % S == 0
        video = F.pad(video.reshape(B, 1, T, C * H * W), (0, 0, 0, pad), "replicate").reshape(
            B, -1, C, H, W
        ) # torch.Size([2, 32, 3, 384, 512])

        # print("pad", pad) # 2 # 0 id sequence % 8 = 0

        # Compute convolutional features for the video or for the current chunk in case of online mode
        # self.fnet(video.reshape(-1, C, H, W)) torch.Size([64, 128, 96, 128])
        fmaps = self.fnet(video.reshape(-1, C, H, W)).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )  # torch.Size([2, 32, 128, 96, 128])

        # print("fmaps.shape", fmaps.shape) # torch.Size([B, T+P, 128, 96, 128])

        # Process and integrate labels into fmaps for the first batch
        # train_label_first_expanded_batch, train_label_last_expanded_batch = self.process_labels(fmaps, train_label_first_batch, train_label_last_batch)
        # fmaps = self.add_labels_to_fmaps(fmaps, train_label_first_batch, train_label_last_batch, train_ltrb_target_second_emb)

        # fmaps = self.add_labels_to_fmaps(fmaps, train_label_first_batch)

        if wLST:
            fmaps = self.add_labels_to_fmaps(fmaps, train_label_first_batch, train_label_last_batch)

        # input()


        # We compute track features
        track_feat = self.get_track_feat(
            fmaps,
            queried_frames - self.online_ind if is_online else queried_frames,
            queried_coords,
        ).repeat(1, S, 1, 1)
        # track_feat torch.Size([2, 1, 4, 128]) -> torch.Size([2, 8, 4, 128])
        # print("track_feat.shape", track_feat.shape) torch.Size([B, 8, qurey_point_num, 128])

        if is_online:
            # We update track features for the current window
            sample_frames = queried_frames[:, None, :, None]  # B 1 N 1
            left = 0 if self.online_ind == 0 else self.online_ind + step
            right = self.online_ind + S
            sample_mask = (sample_frames >= left) & (sample_frames < right)
            if self.online_track_feat is None:
                self.online_track_feat = torch.zeros_like(track_feat, device=device)
            self.online_track_feat += track_feat * sample_mask
            track_feat = self.online_track_feat.clone()
        # We process ((num_windows - 1) * step + S) frames in total, so there are
        # (ceil((T - S) / step) + 1) windows
        num_windows = (T - S + step - 1) // step + 1
        # We process only the current video chunk in the online mode
        indices = [self.online_ind] if is_online else range(0, step * num_windows, step)

        coords_init = queried_coords.reshape(B, 1, N, 2).expand(B, S, N, 2).float()
        vis_init = torch.ones((B, S, N, 1), device=device).float() * 10

        if eva_time:
            print("video chunk s")
            start3.record()

        for ind in indices:
            # We copy over coords and vis for tracks that are queried
            # by the end of the previous window, which is ind + overlap
            if ind > 0:
                overlap = S - step
                copy_over = (queried_frames < ind + overlap)[:, None, :, None]  # B 1 N 1
                coords_prev = torch.nn.functional.pad(
                    coords_predicted[:, ind : ind + overlap] / self.stride,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 2
                vis_prev = torch.nn.functional.pad(
                    vis_predicted[:, ind : ind + overlap, :, None].clone(),
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 1
                coords_init = torch.where(
                    copy_over.expand_as(coords_init), coords_prev, coords_init
                )
                vis_init = torch.where(copy_over.expand_as(vis_init), vis_prev, vis_init)

            # The attention mask is 1 for the spatio-temporal points within
            # a track which is updated in the current window
            attention_mask = (queried_frames < ind + S).reshape(B, 1, N).repeat(1, S, 1)  # B S N

            # The track mask is 1 for the spatio-temporal points that actually
            # need updating: only after begin queried, and not if contained
            # in a previous window
            track_mask = (
                queried_frames[:, None, :, None]
                <= torch.arange(ind, ind + S, device=device)[None, :, None, None]
            ).contiguous()  # B S N 1

            if ind > 0:
                track_mask[:, :overlap, :, :] = False

            if eva_time:
                print("forward_window s")
                start.record()

            # Predict the coordinates and visibility for the current window
            coords, vis = self.forward_window(
                fmaps=fmaps if is_online else fmaps[:, ind : ind + S],
                coords=coords_init,
                SideNetwork_D=SideNetwork_D,
                SideNetwork_U=SideNetwork_U,
                track_feat=attention_mask.unsqueeze(-1) * track_feat,
                vis=vis_init,
                track_mask=track_mask,
                attention_mask=attention_mask,
                iters=iters,
            )

            if eva_time:
                end.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                print("forward_window", start.elapsed_time(end))

            # len(coords) 6
            # vis.shape torch.Size([2, 8, 4])
            # coords[0].shape torch.Size([2, 8, 4, 2])

            S_trimmed = T if is_online else min(T - ind, S)  # accounts for last window duration
            coords_predicted[:, ind : ind + S] = coords[-1][:, :S_trimmed]
            vis_predicted[:, ind : ind + S] = vis[:, :S_trimmed]
            if is_train:
                all_coords_predictions.append([coord[:, :S_trimmed] for coord in coords])
                all_vis_predictions.append(torch.sigmoid(vis[:, :S_trimmed]))

        if eva_time:
            end3.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            print("video chunk", start3.elapsed_time(end3))

        if is_online:
            self.online_ind += step
            self.online_coords_predicted = coords_predicted
            self.online_vis_predicted = vis_predicted
        vis_predicted = torch.sigmoid(vis_predicted)

        if is_train:
            mask = queried_frames[:, None] <= torch.arange(0, T, device=device)[None, :, None]
            train_data = (all_coords_predictions, all_vis_predictions, mask)
        else:
            train_data = None

        return coords_predicted, vis_predicted, train_data


class CoTracker2_ori(nn.Module):
    def __init__(
        self,
        window_len=8,
        stride=4,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
    ):
        super(CoTracker2, self).__init__()
        self.window_len = window_len
        self.stride = stride
        self.hidden_dim = 256
        self.latent_dim = 128
        self.add_space_attn = add_space_attn
        self.fnet = BasicEncoder(output_dim=self.latent_dim)
        self.num_virtual_tracks = num_virtual_tracks
        self.model_resolution = model_resolution
        self.input_dim = 456
        self.updateformer = EfficientUpdateFormer(
            space_depth=6,
            time_depth=6,
            input_dim=self.input_dim,
            hidden_size=384,
            output_dim=self.latent_dim + 2,
            mlp_ratio=4.0,
            add_space_attn=add_space_attn,
            num_virtual_tracks=num_virtual_tracks,
        )

        time_grid = torch.linspace(0, window_len - 1, window_len).reshape(1, window_len, 1)

        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(self.input_dim, time_grid[0])
        )

        self.register_buffer(
            "pos_emb",
            get_2d_sincos_pos_embed(
                embed_dim=self.input_dim,
                grid_size=(
                    model_resolution[0] // stride,
                    model_resolution[1] // stride,
                ),
            ),
        )
        self.norm = nn.GroupNorm(1, self.latent_dim)
        self.track_feat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.vis_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 1),
        )

    def forward_window(
        self,
        fmaps,
        coords,
        track_feat=None,
        vis=None,
        track_mask=None,
        attention_mask=None,
        iters=4,
    ):
        # B = batch size
        # S = number of frames in the window)
        # N = number of tracks
        # C = channels of a point feature vector
        # E = positional embedding size
        # LRR = local receptive field radius
        # D = dimension of the transformer input tokens

        # track_feat = B S N C
        # vis = B S N 1
        # track_mask = B S N 1
        # attention_mask = B S N

        B, S_init, N, __ = track_mask.shape
        B, S, *_ = fmaps.shape

        track_mask = F.pad(track_mask, (0, 0, 0, 0, 0, S - S_init), "constant")
        track_mask_vis = (
            torch.cat([track_mask, vis], dim=-1).permute(0, 2, 1, 3).reshape(B * N, S, 2)
        )

        corr_block = CorrBlock(
            fmaps,
            num_levels=4,
            radius=3,
            padding_mode="border",
        )

        sampled_pos_emb = (
            sample_features4d(self.pos_emb.repeat(B, 1, 1, 1), coords[:, 0])
            .reshape(B * N, self.input_dim)
            .unsqueeze(1)
        )  # B E N -> (B N) 1 E

        coord_preds = []
        for __ in range(iters):
            coords = coords.detach()  # B S N 2
            corr_block.corr(track_feat)

            # Sample correlation features around each point
            fcorrs = corr_block.sample(coords)  # (B N) S LRR

            # Get the flow embeddings
            flows = (coords - coords[:, 0:1]).permute(0, 2, 1, 3).reshape(B * N, S, 2)
            flow_emb = get_2d_embedding(flows, 64, cat_coords=True)  # N S E

            track_feat_ = track_feat.permute(0, 2, 1, 3).reshape(B * N, S, self.latent_dim)

            transformer_input = torch.cat([flow_emb, fcorrs, track_feat_, track_mask_vis], dim=2)
            x = transformer_input + sampled_pos_emb + self.time_emb
            x = x.view(B, N, S, -1)  # (B N) S D -> B N S D

            delta = self.updateformer(
                x,
                attention_mask.reshape(B * S, N),  # B S N -> (B S) N
            )

            delta_coords = delta[..., :2].permute(0, 2, 1, 3)
            coords = coords + delta_coords
            coord_preds.append(coords * self.stride)

            delta_feats_ = delta[..., 2:].reshape(B * N * S, self.latent_dim)
            track_feat_ = track_feat.permute(0, 2, 1, 3).reshape(B * N * S, self.latent_dim)
            track_feat_ = self.track_feat_updater(self.norm(delta_feats_)) + track_feat_
            track_feat = track_feat_.reshape(B, N, S, self.latent_dim).permute(
                0, 2, 1, 3
            )  # (B N S) C -> B S N C # B 8 4 128

        vis_pred = self.vis_predictor(track_feat).reshape(B, S, N)
        return coord_preds, vis_pred
        # len(coord_preds) 6
        # vis_pred.shape torch.Size([2, 8, 4])

    def get_track_feat(self, fmaps, queried_frames, queried_coords):
        sample_frames = queried_frames[:, None, :, None]
        sample_coords = torch.cat(
            [
                sample_frames,
                queried_coords[:, None],
            ],
            dim=-1,
        )
        sample_track_feats = sample_features5d(fmaps, sample_coords)
        return sample_track_feats

    def init_video_online_processing(self):
        self.online_ind = 0
        self.online_track_feat = None
        self.online_coords_predicted = None
        self.online_vis_predicted = None

    def forward(self, video, queries, iters=4, is_train=False, is_online=False):
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3]): input videos.
            queries (FloatTensor[B, N, 3]): point queries.
            iters (int, optional): number of updates. Defaults to 4.
            is_train (bool, optional): enables training mode. Defaults to False.
            is_online (bool, optional): enables online mode. Defaults to False. Before enabling, call model.init_video_online_processing().

        Returns:
            - coords_predicted (FloatTensor[B, T, N, 2]):
            - vis_predicted (FloatTensor[B, T, N]):
            - train_data: `None` if `is_train` is false, otherwise:
                - all_vis_predictions (List[FloatTensor[B, S, N, 1]]):
                - all_coords_predictions (List[FloatTensor[B, S, N, 2]]):
                - mask (BoolTensor[B, T, N]):
        """
        B, T, C, H, W = video.shape
        B, N, __ = queries.shape
        S = self.window_len
        device = queries.device

        # B = batch size
        # S = number of frames in the window of the padded video
        # S_trimmed = actual number of frames in the window
        # N = number of tracks
        # C = color channels (3 for RGB)
        # E = positional embedding size
        # LRR = local receptive field radius
        # D = dimension of the transformer input tokens

        # video = B T C H W
        # queries = B N 3
        # coords_init = B S N 2
        # vis_init = B S N 1

        assert S >= 2  # A tracker needs at least two frames to track something
        if is_online:
            assert T <= S, "Online mode: video chunk must be <= window size."
            assert self.online_ind is not None, "Call model.init_video_online_processing() first."
            assert not is_train, "Training not supported in online mode."
        step = S // 2  # How much the sliding window moves at every step
        video = 2 * (video / 255.0) - 1.0

        # The first channel is the frame number
        # The rest are the coordinates of points we want to track
        queried_frames = queries[:, :, 0].long()

        queried_coords = queries[..., 1:]
        queried_coords = queried_coords / self.stride

        # We store our predictions here
        coords_predicted = torch.zeros((B, T, N, 2), device=device)
        vis_predicted = torch.zeros((B, T, N), device=device)
        if is_online:
            if self.online_coords_predicted is None:
                # Init online predictions with zeros
                self.online_coords_predicted = coords_predicted
                self.online_vis_predicted = vis_predicted
            else:
                # Pad online predictions with zeros for the current window
                pad = min(step, T - step)
                coords_predicted = F.pad(
                    self.online_coords_predicted, (0, 0, 0, 0, 0, pad), "constant"
                )
                vis_predicted = F.pad(self.online_vis_predicted, (0, 0, 0, pad), "constant")
        all_coords_predictions, all_vis_predictions = [], []

        # Pad the video so that an integer number of sliding windows fit into it
        # TODO: we may drop this requirement because the transformer should not care
        # TODO: pad the features instead of the video
        # pad 2 T 30
        pad = S - T if is_online else (S - T % S) % S  # We don't want to pad if T % S == 0
        video = F.pad(video.reshape(B, 1, T, C * H * W), (0, 0, 0, pad), "replicate").reshape(
            B, -1, C, H, W
        ) # torch.Size([2, 32, 3, 384, 512])

        # Compute convolutional features for the video or for the current chunk in case of online mode
        # self.fnet(video.reshape(-1, C, H, W)) torch.Size([64, 128, 96, 128])
        fmaps = self.fnet(video.reshape(-1, C, H, W)).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )  # torch.Size([2, 32, 128, 96, 128])

        # We compute track features
        track_feat = self.get_track_feat(
            fmaps,
            queried_frames - self.online_ind if is_online else queried_frames,
            queried_coords,
        ).repeat(1, S, 1, 1)
        # track_feat torch.Size([2, 1, 4, 128]) -> torch.Size([2, 8, 4, 128])

        if is_online:
            # We update track features for the current window
            sample_frames = queried_frames[:, None, :, None]  # B 1 N 1
            left = 0 if self.online_ind == 0 else self.online_ind + step
            right = self.online_ind + S
            sample_mask = (sample_frames >= left) & (sample_frames < right)
            if self.online_track_feat is None:
                self.online_track_feat = torch.zeros_like(track_feat, device=device)
            self.online_track_feat += track_feat * sample_mask
            track_feat = self.online_track_feat.clone()
        # We process ((num_windows - 1) * step + S) frames in total, so there are
        # (ceil((T - S) / step) + 1) windows
        num_windows = (T - S + step - 1) // step + 1
        # We process only the current video chunk in the online mode
        indices = [self.online_ind] if is_online else range(0, step * num_windows, step)

        coords_init = queried_coords.reshape(B, 1, N, 2).expand(B, S, N, 2).float()
        vis_init = torch.ones((B, S, N, 1), device=device).float() * 10
        for ind in indices:
            # We copy over coords and vis for tracks that are queried
            # by the end of the previous window, which is ind + overlap
            if ind > 0:
                overlap = S - step
                copy_over = (queried_frames < ind + overlap)[:, None, :, None]  # B 1 N 1
                coords_prev = torch.nn.functional.pad(
                    coords_predicted[:, ind : ind + overlap] / self.stride,
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 2
                vis_prev = torch.nn.functional.pad(
                    vis_predicted[:, ind : ind + overlap, :, None].clone(),
                    (0, 0, 0, 0, 0, step),
                    "replicate",
                )  # B S N 1
                coords_init = torch.where(
                    copy_over.expand_as(coords_init), coords_prev, coords_init
                )
                vis_init = torch.where(copy_over.expand_as(vis_init), vis_prev, vis_init)

            # The attention mask is 1 for the spatio-temporal points within
            # a track which is updated in the current window
            attention_mask = (queried_frames < ind + S).reshape(B, 1, N).repeat(1, S, 1)  # B S N

            # The track mask is 1 for the spatio-temporal points that actually
            # need updating: only after begin queried, and not if contained
            # in a previous window
            track_mask = (
                queried_frames[:, None, :, None]
                <= torch.arange(ind, ind + S, device=device)[None, :, None, None]
            ).contiguous()  # B S N 1

            if ind > 0:
                track_mask[:, :overlap, :, :] = False

            # Predict the coordinates and visibility for the current window
            coords, vis = self.forward_window(
                fmaps=fmaps if is_online else fmaps[:, ind : ind + S],
                coords=coords_init,
                track_feat=attention_mask.unsqueeze(-1) * track_feat,
                vis=vis_init,
                track_mask=track_mask,
                attention_mask=attention_mask,
                iters=iters,
            )
            # len(coords) 6
            # vis.shape torch.Size([2, 8, 4])
            # coords[0].shape torch.Size([2, 8, 4, 2])

            S_trimmed = T if is_online else min(T - ind, S)  # accounts for last window duration
            coords_predicted[:, ind : ind + S] = coords[-1][:, :S_trimmed]
            vis_predicted[:, ind : ind + S] = vis[:, :S_trimmed]
            if is_train:
                all_coords_predictions.append([coord[:, :S_trimmed] for coord in coords])
                all_vis_predictions.append(torch.sigmoid(vis[:, :S_trimmed]))

        if is_online:
            self.online_ind += step
            self.online_coords_predicted = coords_predicted
            self.online_vis_predicted = vis_predicted
        vis_predicted = torch.sigmoid(vis_predicted)

        if is_train:
            mask = queried_frames[:, None] <= torch.arange(0, T, device=device)[None, :, None]
            train_data = (all_coords_predictions, all_vis_predictions, mask)
        else:
            train_data = None

        return coords_predicted, vis_predicted, train_data


class EfficientUpdateFormer(nn.Module):
    """
    Transformer model that updates track estimates.
    """

    def __init__(
        self,
        space_depth=6,
        time_depth=6,
        input_dim=320,
        hidden_size=384,
        num_heads=8,
        output_dim=130,
        mlp_ratio=4.0,
        add_space_attn=True,
        num_virtual_tracks=64,
    ):
        super().__init__()
        self.out_channels = 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.flow_head = torch.nn.Linear(hidden_size, output_dim, bias=True)
        self.num_virtual_tracks = num_virtual_tracks
        self.virual_tracks = nn.Parameter(torch.randn(1, num_virtual_tracks, 1, hidden_size))
        self.time_blocks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                )
                for _ in range(time_depth)
            ]
        )

        if add_space_attn:
            self.space_virtual_blocks = nn.ModuleList(
                [
                    AttnBlock(
                        hidden_size,
                        num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_class=Attention,
                    )
                    for _ in range(space_depth)
                ]
            )
            self.space_point2virtual_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio)
                    for _ in range(space_depth)
                ]
            )
            self.space_virtual2point_blocks = nn.ModuleList(
                [
                    CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio)
                    for _ in range(space_depth)
                ]
            )
            assert len(self.time_blocks) >= len(self.space_virtual2point_blocks)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, input_tensor, mask=None):
        tokens = self.input_transform(input_tensor)
        B, _, T, _ = tokens.shape
        virtual_tokens = self.virual_tracks.repeat(B, 1, T, 1)
        tokens = torch.cat([tokens, virtual_tokens], dim=1)
        _, N, _, _ = tokens.shape

        j = 0
        for i in range(len(self.time_blocks)):
            time_tokens = tokens.contiguous().view(B * N, T, -1)  # B N T C -> (B N) T C
            time_tokens = self.time_blocks[i](time_tokens)

            tokens = time_tokens.view(B, N, T, -1)  # (B N) T C -> B N T C
            if self.add_space_attn and (
                i % (len(self.time_blocks) // len(self.space_virtual_blocks)) == 0
            ):
                space_tokens = (
                    tokens.permute(0, 2, 1, 3).contiguous().view(B * T, N, -1)
                )  # B N T C -> (B T) N C
                point_tokens = space_tokens[:, : N - self.num_virtual_tracks]
                virtual_tokens = space_tokens[:, N - self.num_virtual_tracks :]

                virtual_tokens = self.space_virtual2point_blocks[j](
                    virtual_tokens, point_tokens, mask=mask
                )
                virtual_tokens = self.space_virtual_blocks[j](virtual_tokens)
                point_tokens = self.space_point2virtual_blocks[j](
                    point_tokens, virtual_tokens, mask=mask
                )
                space_tokens = torch.cat([point_tokens, virtual_tokens], dim=1)
                tokens = space_tokens.view(B, T, N, -1).permute(0, 2, 1, 3)  # (B T) N C -> B N T C
                j += 1
        tokens = tokens[:, : N - self.num_virtual_tracks]
        flow = self.flow_head(tokens)
        return flow


class CrossAttnBlock(nn.Module):
    def __init__(self, hidden_size, context_dim, num_heads=1, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(hidden_size)
        self.cross_attn = Attention(
            hidden_size, context_dim=context_dim, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, context, mask=None):
        if mask is not None:
            if mask.shape[1] == x.shape[1]:
                mask = mask[:, None, :, None].expand(
                    -1, self.cross_attn.heads, -1, context.shape[1]
                )
            else:
                mask = mask[:, None, None].expand(-1, self.cross_attn.heads, x.shape[1], -1)

            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value
        x = x + self.cross_attn(
            self.norm1(x), context=self.norm_context(context), attn_bias=attn_bias
        )
        x = x + self.mlp(self.norm2(x))
        return x
