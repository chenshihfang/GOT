import os
import torch

from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
# from IPython.display import HTML
import cv2
from torch.nn import DataParallel

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# video1 = read_video_from_path('./assets/avist_balloon_fight.mp4')
# video1 = read_video_from_path('./assets/avist_beyblade_competition.mp4')
# video1 = read_video_from_path('./assets/avist_car_in_smoke_night.mp4')
# video1 = read_video_from_path('./assets/avist_pygmy_seahorse_2.mp4')
video1 = read_video_from_path('./assets/avist_ducklings_stairs.mp4')
# video1 = read_video_from_path('./assets/avist_stick_insect_1.mp4')
# video1 = read_video_from_path('./assets/avist_stick_insect_2.mp4')

# video1 = read_video_from_path('./assets/Avist_104_short.mp4')
# video1 = read_video_from_path('./assets/avist_flying_bees_2.mp4')
# video = read_video_from_path('./assets/avist_ambulance_in_night_1.webm')
# video = read_video_from_path('./assets/Avist_114.mp4')
# video2 = read_video_from_path('./assets/Avist_114_half.mp4')
# video2 = read_video_from_path('./assets/avist_ducklings_stairs_short.mp4')

video = torch.from_numpy(video1).permute(0, 3, 1, 2)[None].float()
# video1 = torch.from_numpy(video1).permute(0, 3, 1, 2)[None].float()

# video2 = torch.from_numpy(video2).permute(0, 3, 1, 2)[None].float()


###
# Initialize an empty list to store queries
queries = []

# # Define the callback function for mouse events
# def callback(event, x, y, flags, param):
#     global queries
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # Append [frame_index, x, y] to the queries list
#         queries.append([0, x, y])
#         # Draw a point at the clicked position
#         cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
#         cv2.namedWindow('First Frame', cv2.WINDOW_NORMAL)
#         cv2.imshow('First Frame', img)

# # Convert the first frame of your video tensor to numpy array and then to uint8
# first_frame = video[0, 0].permute(1, 2, 0).numpy().astype('uint8')

# # Make a copy of the first frame to display
# img = first_frame.copy()

# # Create a window and set the callback function for mouse events
# cv2.namedWindow('First Frame')
# cv2.namedWindow('First Frame', cv2.WINDOW_NORMAL)
# cv2.setMouseCallback('First Frame', callback)

# # Show the first frame
# cv2.imshow('First Frame', img)

# # Wait until 'Enter' is pressed
# while True:
#     if cv2.waitKey(1) & 0xFF == ord('\r'):  # '\r' is the Enter key
#         break

# # Destroy the window
# cv2.destroyAllWindows()

# Convert queries to a PyTorch tensor
queries = torch.tensor(queries).float()

# print("Collected queries:", queries)

# input()
####
from cotracker.predictor import CoTrackerPredictor

# model = CoTrackerPredictor(
#     checkpoint=os.path.join(
#         './checkpoints/cotracker_stride_4_wind_8.pth'
#         # './checkpoints/cotracker_stride_4_wind_12.pth'
#         # './checkpoints/cotracker_stride_8_wind_16.pth'

#     )
# )

video = video.squeeze(0)[0:90].unsqueeze(0)

if torch.cuda.is_available():
    # model = model.cuda()
    # model = DataParallel(model, device_ids=[0,1])  # Wrap model in DataParallel
    # model = DataParallel(model, device_ids=[0])  # Wrap model in DataParallel

    video = video.cuda()
    # video1 = video1.cuda()
    # video2 = video2.cuda()

# videos = video1
# print("video1.shape", video1.shape)

# video2 = video2.squeeze(0)[0:30].unsqueeze(0)

# print("video2.shape", video2.shape)

# video = torch.cat((video1, video2), 0)
# video = torch.cat((video, video), 0)
# video = torch.cat((video2, video2), 0)

video = video.expand(2,-1,-1,-1,-1)
print("video.shape", video.shape)
# input()
### method 1
# pred_tracks, pred_visibility = model(video, grid_size=20)

# # vis = Visualizer(save_dir='./videos', pad_value=100)
# vis = Visualizer(save_dir='./videos')
# vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename='teaser')

### method 2

queries = torch.tensor([
    [0., 674., 160.],  # point tracked from the first frame
    [0., 674., 170.],  # point tracked from the first frame
    [0., 674., 180.],  # point tracked from the first frame
])


queries2 = torch.tensor([
    [0., 222., 222.],  # point tracked from the first frame
    [0., 111., 111.],  # point tracked from the first frame
    [0., 333., 333.],  # point tracked from the first frame

])

# queries2 = torch.tensor([
#     [0., 111., 111.],  # point tracked from the first frame
#     [0., 222., 222.],  # point tracked from the first frame

# ])

# avist_pygmy_seahorse_2
# queries = torch.tensor([
#     [0., 750., 320.],  # point tracked from the first frame
#     [0., 750., 325.], # frame number 10
#     [0., 800., 320.], # ...
#     [0., 800., 325.],

#     # [5., 100., 100.],  # point tracked from the first frame
#     # [10., 200., 200.], # frame number 10
#     # [15., 300., 300.], # ...
#     # [20., 400., 300.],
# ])

# avist_ambulance_in_night_1
# queries2 = torch.tensor([
#     [0., 700., 500.], # ...
#     [0., 700., 550.], # ...
#     [0., 700., 600.],

#     [0., 750., 500.], # ...
#     [0., 750., 550.], # ...
#     [0., 750., 600.],
# ])

# Avist_114
# queries = torch.tensor([
#     [0., 600., 400.], # ...
#     [0., 650., 450.], # ...
#     [0., 700., 500.],

#     [0., 750., 400.], # ...
#     [0., 800., 450.], # ...
#     [0., 850., 500.],
# ])


# avist_ducklings_stairs_short
# queries2 = torch.tensor([

#     [0., 544., 459.], # ...
#     [0., 550., 500.], # ...
#     [0., 592., 503.],

#     [0., 644., 530.], # ...
#     # [0., 685., 516.],


#     # [0., 651., 484.], # ...
#     # [0., 356., 583.], # ...
#     # [0., 430., 570.],


# ])

queries = queries[None]
queries2 = queries2[None]
# queries2 = queries2[None]
queries = torch.cat((queries, queries2), 0)

if torch.cuda.is_available():
    queries = queries.cuda()

# print("queries.shape", queries.shape)

# queries = queries.expand(2,-1,-1)
print("queries", queries)
print("queries.shape", queries.shape)
# print("None", None)

# print("queries[None].shape", queries[None].shape)

# input()


print("start pred")


# pred_tracks, pred_visibility = model(video, queries=queries[None])
# pred_tracks, pred_visibility = model(video, queries=queries)
# pred_tracks, pred_visibility = model(video, queries=queries, backward_tracking=True)

# Run Offline CoTracker:
device = 'cuda'

# cotracker = CoTrackerPredictor(checkpoint="./checkpoints/cotracker2.pth").to(device)

cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)
cotracker = DataParallel(cotracker, device_ids=[0,1])  # Wrap model in DataParallel
grid_size = 100
# pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1
pred_tracks, pred_visibility = cotracker(video, queries=queries, grid_size=grid_size, backward_tracking=True) # B T N 2,  B T N 1

# pred_tracks, pred_visibility = cotracker(video, queries=queries, backward_tracking=True) # B T N 2,  B T N 1

print("end pred")


print("pred_tracks", pred_tracks)

# print("pred_visibility", pred_visibility)


print("pred_tracks.shape", pred_tracks.shape)
# print("pred_visibility.shape", pred_visibility.shape)

# print("pred_tracks[0][0:5]", pred_tracks[0][0:5])
# input()


vis = Visualizer(
    save_dir='./saved_videos',
    linewidth=2,
    mode='cool',
    tracks_leave_trace=0
)

# vis.visualize(
#     video=video,
#     tracks=pred_tracks,
#     visibility=pred_visibility,
#     filename='avist_flying_bees_2_cotracker')

vis.visualize(
    video=video,
    tracks=pred_tracks[0][None],
    visibility=pred_visibility[0][None],
    filename='video1')

vis.visualize(
    video=video,
    tracks=pred_tracks[1][None],
    visibility=pred_visibility[1][None],
    filename='video2')

print("end")