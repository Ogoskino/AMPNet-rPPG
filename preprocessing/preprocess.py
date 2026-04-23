import torch
import numpy as np
from preprocessing.dataloader import *



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_array(arr):
    """
    Normalize a NumPy array to the range [0, 1].

    Parameters:
    arr (np.ndarray): The input array to normalize.

    Returns:
    np.ndarray: The normalized array with values between 0 and 1.
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr


def preprocess_iBVP_data(rgb_face, thermal_face, label):
    thermal_array = np.expand_dims(thermal_face, axis=-1)
    rgb_faces = np.array(rgb_face)/rgb_face.max()
    thermal_array = np.array(thermal_array)/thermal_array.max()
    videos = np.concatenate((rgb_faces, thermal_array), axis=-1)
    bvps = label
    a = np.array(videos)
    b = normalize_array(np.array(bvps))
    return a, b

def preprocess_PURE_data(rgb_face, label):
    videos = np.array(rgb_face)/rgb_face.max()
    bvps = label
    a = np.array(videos)
    b = normalize_array(np.array(bvps))
    return a, b

def extract_segments(videos, bvps, sequence_length=192):

    n_videos, total_frames, height, width, num_channels = videos.shape
    assert (n_videos*total_frames) % sequence_length == 0, "Total frames must be divisible by sequence length."
    
    # Define reshape shapes
    video_shape = (-1, num_channels, sequence_length, height, width)
    label_shape = (-1, sequence_length)
    
    # Reshape and convert to tensors
    video_chunks = torch.tensor(videos.reshape(video_shape), dtype=torch.float32)
    label_chunks = torch.tensor(bvps.reshape(label_shape), dtype=torch.float32)
    
    return video_chunks, label_chunks





if __name__ == "__main__":

    data = 'PURE'

    if data == 'iBVP':
        dataset_path = r'C:\Users\jkogo\OneDrive\Desktop\PHD resources\datasets\iBVP_Dataset'
        rgb_face, thermal_face, label = load_iBVP_dataset(dataset_path)
        a, b = preprocess_iBVP_data(rgb_face, thermal_face, label)
        video_chunks, label_chunks = extract_segments(a, b)
        print(video_chunks.shape)  # Should be [num_chunks, chunk_size, height, width, channels]
        print(label_chunks.shape)  # Should be [num_chunks, chunk_size]

    else:
        #base_path = r'C:\Users\n1071552\Desktop\Pure'
        #videos_pure, bvps_pure = extract_PURE_videos_and_bvps(base_path)
        videos_pure, bvps_pure = np.random.rand(12, 1792, 64, 64, 3), np.random.rand(12, 1792)
        a_pure, b_pure = preprocess_PURE_data(videos_pure, bvps_pure)
        video_chunks_pure, label_chunks_pure = extract_segments(a_pure, b_pure)
        print(video_chunks_pure.shape)  # Should be [num_chunks, chunk_size, height, width, channels]
        print(label_chunks_pure.shape)  # Should be [num_chunks, chunk_size]