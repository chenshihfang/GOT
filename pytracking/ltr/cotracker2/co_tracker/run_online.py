
import os
import torch
from base64 import b64encode
import cv2
from cotracker.utils.visualizer import Visualizer, read_video_from_path

device = 'cuda'
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online").to(device)


grid_size = 30
video1 = read_video_from_path('./assets/avist_flying_bees_2.mp4')
video = torch.from_numpy(video1).permute(0, 3, 1, 2)[None].float()
video = video.squeeze(0)[0:270].unsqueeze(0)
video = video.cuda()

# Run Online CoTracker, the same model with a different API:
# Initialize online processing
cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)  

# Process the video
for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
    pred_tracks, pred_visibility = cotracker(
        video_chunk=video[:, ind : ind + cotracker.step * 2]
    )  # B T N 2,  B T N 1


vis = Visualizer(
    save_dir='./videos',
    linewidth=2,
    mode='cool',
    tracks_leave_trace=0
)

print("pred_tracks.shape", pred_tracks.shape)

vis.visualize(
    video=video,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename='avist_avist_ducklings_stairs_cotracker')