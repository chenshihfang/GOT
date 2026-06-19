import torch
import numpy as np
from torch.nn import DataParallel
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor
import time
from cotracker.utils.visualizer import Visualizer, read_video_from_path

def run_cotracker(video, bb_first, cotracker, num_points=9, grid_size=100, frames_num=30):
    if torch.cuda.is_available():
        video = video.cuda()
    print("video shape:", video.shape)  # torch.Size([6, 30, 3, 720, 1280])
    
    # List to store the sampled queries
    sampled_queries_list = []

    # Ensure bb_first has shape [images_num, batch, 4]
    if bb_first.dim() == 2:  # If bb_first is [batch, 4], unsqueeze to [1, batch, 4]
        bb_first = bb_first.unsqueeze(0)
    
    # Reshape bb_first for easier processing
    # bb_first_flat shape: [total_boxes, 4]
    bb_first_flat = bb_first.reshape(-1, 4)

    # Iterate over each box in the flattened bb_first
    for box in bb_first_flat:
        x, y, w, h = box
        # Generate random points within the box
        sampled_x = x + torch.rand(num_points) * w
        sampled_y = y + torch.rand(num_points) * h
        # Concatenate the frame index (0) with the sampled points
        sampled_queries = torch.cat((torch.zeros(num_points, 1), sampled_x.unsqueeze(-1), sampled_y.unsqueeze(-1)), dim=1)
        sampled_queries_list.append(sampled_queries)
    print("sampled_queries_list length:", len(sampled_queries_list)) # 6

    # Convert the list to a tensor and reshape to match the original bb_first shape
    # sampled_queries shape: [images_num, batch, num_points, 3]
    sampled_queries = torch.stack(sampled_queries_list).reshape(bb_first.shape[0], -1, num_points, 3)
    print("sampled_queries shape:", sampled_queries.shape) # torch.Size([2, 3, 9, 3])

    if torch.cuda.is_available():
        sampled_queries = sampled_queries.cuda()

    # Run CoTracker prediction
    start_pred_time = time.time()
    pred_tracks, pred_visibility = cotracker(video, queries=sampled_queries.view(-1, num_points, 3), grid_size=grid_size, backward_tracking=True)
    end_pred_time = time.time()

    # Reshape the predicted tracks and visibility tensors
    # pred_tracks shape: [images_num, batch, frames_num, num_points, 2]
    # pred_visibility shape: [images_num, batch, frames_num, num_points]
    pred_tracks = pred_tracks.view(bb_first.shape[0], -1, frames_num, num_points, 2)
    pred_visibility = pred_visibility.view(bb_first.shape[0], -1, frames_num, num_points)

    print("Predicted tracks shape:", pred_tracks.shape) # torch.Size([2, 3, 30, 9, 2])
    print("Predicted visibility shape:", pred_visibility.shape) # torch.Size([2, 3, 30, 9])
    print("CoTracker prediction time:", end_pred_time - start_pred_time, "seconds")

    ###############################################################
    
    # Average over the first, third, and fourth dimensions
    avg_pred_tracks = pred_tracks.mean(dim=(0, 2, 3))

    # Sum over the last dimension to get the final tensor with shape [3, 1]
    final_pred_tracks = avg_pred_tracks.sum(dim=-1, keepdim=True)
    print("final_pred_tracks shape:", final_pred_tracks.shape) # torch.Size([3, 1])

    # Define B based on the shape of pred_tracks
    B = pred_tracks.shape[1]

    # Reshape final_pred_tracks to [B, 1, 1, 1] for broadcasting
    final_pred_tracks_reshaped = final_pred_tracks.view(B, 1, 1, 1)
    print("final_pred_tracks_reshaped shape:", final_pred_tracks_reshaped.shape) # torch.Size([3, 1, 1, 1])

    ###############################################################

    return pred_tracks, pred_visibility, final_pred_tracks_reshaped

start_time = time.time()

# Set the number of images and batches
# images_num = 2
# batch = 3
images_num = 2
batch = 3
frames_num = 16
# frames_num = 60

# Example train_imgs_all tensor with shape [frames_num*images_num, batch, 3, H, W]
video_path = "/home/sfchen94/pytrackingcsf/pytracking/ltr/cotracker2/co_tracker/assets/avist_ducklings_stairs.mp4"
video = read_video_from_path(video_path)
# video = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float()[:, :60]
# video = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float()[:, :frames_num]


# Assuming 'video' is a numpy array with shape (num_frames, height, width, channels)
# Ensure frames_num is <= total frames available divided by stride (10 in this case)
stride = 5
total_frames = video.shape[0]

# Calculate the frame indices you want to sample
sample_indices = torch.arange(0, frames_num * stride, step=stride)
sample_indices = sample_indices[sample_indices < total_frames]  # Ensure within bounds

# Convert video to a PyTorch tensor, select sampled frames, and reshape
video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)  # Shape: (num_frames, channels, height, width)
sampled_frames = video_tensor[sample_indices]  # Sample frames at stride intervals
video = sampled_frames.unsqueeze(0).float()  # Add batch dimension

# Check final shape (should be (1, frames_num, channels, height, width))
print(video.shape)

H, W = video.shape[-2:]

# Create train_imgs_all by replicating the video tensor
train_imgs_all = video.expand(images_num * batch, frames_num, 3, H, W)
print("train_imgs_all shape:", train_imgs_all.shape) # torch.Size([30, 3, 3, 720, 1280] or [8, 3, 3, 720, 1280])

# Example train_bb_first tensor with shape [images_num, batch, 4]

train_bb_first = torch.tensor([
    [[570.5671, 480.2090, 40.8657, 40.2239],
     [570.5671, 480.2090, 40.8657, 40.2239],
     [570.5671, 480.2090, 40.8657, 40.2239]],
    [[570.5671, 480.2090, 40.8657, 40.2239],
     [570.5671, 480.2090, 40.8657, 40.2239],
     [570.5671, 480.2090, 40.8657, 40.2239]]
])


print("train_bb_first shape:", train_bb_first.shape) # torch.Size([2, 3, 4])

# If images_num is 1, use only the first element of train_bb_first
if images_num == 1:
    train_bb_first = train_bb_first[0].unsqueeze(0)


# Load CoTracker model
device = 'cuda'
cotracker = CoTrackerPredictor(checkpoint="./checkpoints/cotracker2.pth").to(device)
cotracker = DataParallel(cotracker, device_ids=[0])  # Wrap model in DataParallel

# Number of points to sample
num_points = 100

# Run CoTracker prediction
pred_tracks, pred_visibility, final_pred_tracks_reshaped = run_cotracker(train_imgs_all.view(images_num * batch, frames_num, 3, H, W), train_bb_first, cotracker, num_points=num_points, frames_num=frames_num)

# Initialize visualizer and save tracking results
vis = Visualizer(save_dir='./saved_videos', linewidth=2, mode='cool', tracks_leave_trace=0)
video_index = 0
for i in range(images_num):
    for j in range(batch):
        video_ij = train_imgs_all.view(images_num * batch, frames_num, 3, H, W)[video_index]
        tracks_ij = pred_tracks.view(images_num * batch, frames_num, num_points, 2)[video_index]  # Shape: [30, 9, 2]
        visibility_ij = pred_visibility.view(images_num * batch, frames_num, num_points)[video_index]  # Shape: [30, 9]
        filename_ij = f'video{video_index + 1}'
        vis.visualize(video=video_ij[None], tracks=tracks_ij[None], visibility=visibility_ij[None], filename=filename_ij)
        video_index += 1


end_time = time.time()
print("Total execution time:", end_time - start_time, "seconds")
