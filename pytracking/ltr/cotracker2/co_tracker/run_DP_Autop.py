import torch
import time
from torch.nn import DataParallel
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

start_time = time.time()

# Read video and convert to tensor
# video_path = './assets/avist_ducklings_stairs.mp4'
video_path = '/home/sfchen94/pytrackingcsf/cotracker2/co-tracker/assets/avist_ducklings_stairs.mp4'
video = read_video_from_path(video_path)
video = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float()
print("Video shape:", video.shape) # torch.Size([1, 1274, 3, 720, 1280])


# Define query points for tracking
queries_list = [
    torch.tensor([
        [0., 674., 160.],
        [0., 674., 170.],
        [0., 674., 180.],
        [0., 112., 112.]
    ]),
    torch.tensor([
        [0., 222., 222.],
        [0., 111., 111.],
        [0., 333., 333.],
        [0., 112., 112.]
    ]),
    # torch.tensor([
    #     [0., 555., 222.],
    #     [0., 555., 111.],
    #     [0., 555., 333.],
    #     [0., 112., 112.]
    # ]),
    # torch.tensor([
    #     [0., 555., 222.],
    #     [0., 555., 111.],
    #     [0., 555., 333.],
    #     [0., 112., 112.]
    # ])
]


# Select a subset of frames and expand for batch processing
# video = video[:, :90].expand(len(queries_list), -1, -1, -1, -1)
video = video[:, :30].expand(len(queries_list), -1, -1, -1, -1)

print("Processed video shape:", video.shape) # torch.Size([2, 30, 3, 720, 1280])

if torch.cuda.is_available():
    video = video.cuda()

# Rearrange the dimensions (fames_num, batch, channel, H, W) -> (batch, fames_num, channel, H, W)
# video = video.permute(1, 0, 2, 3, 4)  # New shape [3, 10, 3, 432, 432]

# Concatenate all queries into a single tensor
queries = torch.cat([q[None] for q in queries_list], 0)
print("Queries shape:", queries.shape) # torch.Size([2, 4, 3])
if torch.cuda.is_available():
    queries = queries.cuda()

##############################################################################

# Load CoTracker model
device = 'cuda'

# cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)
cotracker = CoTrackerPredictor(checkpoint="./checkpoints/cotracker2.pth").to(device)


# cotracker = DataParallel(cotracker)
cotracker = DataParallel(cotracker, device_ids=[0])  # Wrap model in DataParallel

# Run CoTracker prediction

start_pred_time = time.time()

# pred_tracks, pred_visibility = cotracker(video, queries=queries, backward_tracking=True)

grid_size = 100
pred_tracks, pred_visibility = cotracker(video, queries=queries, grid_size=grid_size, backward_tracking=True)
# pred_tracks, pred_visibility = cotracker(video, queries=queries, grid_size=grid_size)

end_pred_time = time.time()
print("Predicted tracks shape:", pred_tracks.shape) # torch.Size([2, 30, 4, 2])
print("Predicted visibility shape:", pred_visibility.shape) # torch.Size([2, 30, 4])
print("CoTracker prediction time:", end_pred_time - start_pred_time, "seconds")

# Initialize visualizer and save tracking results
# vis = Visualizer(save_dir='./saved_videos', linewidth=2, mode='cool', tracks_leave_trace=0)
# for i in range(len(queries_list)):
#     vis.visualize(video=video, tracks=pred_tracks[i][None], visibility=pred_visibility[i][None], filename=f'video{i+1}')

# end_time = time.time()
# print("Total execution time:", end_time - start_time, "seconds")
