import random
import numpy as np
import os
import mediapipe as mp
import pandas as pd
import cv2
import logging
import glob
import json

logger = logging.basicConfig(filename="rppg.app", filemode='w', level=logging.DEBUG)

def sample_images(rgb_faces, thermal_faces, labels, target_length=5376):
    # Get the number of images in each modality
    num_rgb = len(rgb_faces)
    num_thermal = len(thermal_faces)
    
    # Determine the minimum number of images
    min_length = min(num_rgb, num_thermal, len(labels))
    
    # If the minimum length is greater than the target length, sample down to target length
    if min_length > target_length:
        sampled_indices = sorted(random.sample(range(min_length), target_length))
    else:
        # If already less than or equal to target length, use all images up to min_length
        sampled_indices = list(range(min_length))
    
    # Sample the images and labels based on the sampled indices
    rgb_faces_sampled = [rgb_faces[i] for i in sampled_indices]
    thermal_faces_sampled = [thermal_faces[i] for i in sampled_indices]
    labels_sampled = [labels[i] for i in sampled_indices]
    
    return rgb_faces_sampled, thermal_faces_sampled, labels_sampled



def load_iBVP_dataset(dataset_path, target_length=5376):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    all_rgb_faces = []
    all_thermal_faces = []
    all_labels = []

    thermal_img_width = 640
    thermal_img_height = 512

    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)
        
        if not os.path.isdir(person_path):
            continue
        
        rgb_folder = os.path.join(person_path, f"{person_folder}_rgb")
        thermal_folder = os.path.join(person_path, f"{person_folder}_t")
        csv_file = os.path.join(person_path, f"{person_folder}_bvp.csv")

        # Load the BVP data
        bvp_data = pd.read_csv(csv_file)
        labels = bvp_data.iloc[:, 0].values  # Extract the first column

        rgb_faces = []
        thermal_faces = []
        
        for img_name in sorted(os.listdir(rgb_folder)):
            img_path = os.path.join(rgb_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Convert the image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Perform face detection
            results = face_detection.process(img_rgb)

            if results.detections:
                # Get the first detected face
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                face = img_rgb[y:y+h, x:x+w]

                # Resize the face to 64x64
                face_resized = cv2.resize(face, (64, 64))
                rgb_faces.append(face_resized)
            else:
                # If no face detected, use a placeholder (black image)
                rgb_faces.append(np.zeros((64, 64, 3), dtype=np.uint8))

        for img_name in sorted(os.listdir(thermal_folder)):
            img_path = os.path.join(thermal_folder, img_name)
            
            # Read and process the thermal .raw file
            thermal_matrix = np.fromfile(img_path, dtype=np.uint16, count=thermal_img_width * thermal_img_height).reshape(thermal_img_height, thermal_img_width)
            thermal_matrix = thermal_matrix.astype(np.float32)
            thermal_matrix = (thermal_matrix * 0.04) - 273.15  # Convert to Celsius

            # Normalize the thermal matrix for visualization purposes (optional)
            thermal_matrix_normalized = cv2.normalize(thermal_matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Perform face detection on normalized thermal image
            thermal_img_colormap = cv2.applyColorMap(thermal_matrix_normalized, cv2.COLORMAP_JET)
            thermal_img_rgb = cv2.cvtColor(thermal_img_colormap, cv2.COLOR_BGR2RGB)
            results = face_detection.process(thermal_img_rgb)

            if results.detections:
                # Get the first detected face
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = thermal_img_rgb.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                face_temp = thermal_matrix[y:y+h, x:x+w]

                # Resize the face to 64x64
                face_temp_resized = cv2.resize(face_temp, (64, 64))
                thermal_faces.append(face_temp_resized)
            else:
                # If no face detected, use a placeholder (zeros matrix)
                thermal_faces.append(np.zeros((64, 64), dtype=np.float32))

        # Ensure consistent length by sampling
        rgb_faces_sampled, thermal_faces_sampled, labels_sampled = sample_images(rgb_faces, thermal_faces, labels, target_length=target_length)

        all_rgb_faces.append(rgb_faces_sampled)
        all_thermal_faces.append(thermal_faces_sampled)
        all_labels.append(labels_sampled)

    # Convert lists to NumPy arrays
    all_rgb_faces_np = np.array(all_rgb_faces)
    all_thermal_faces_np = np.array(all_thermal_faces)
    all_labels_np = np.array(all_labels)

    return all_rgb_faces_np, all_thermal_faces_np, all_labels_np


def read_video(video_file):
    """Reads a video file, detects faces, and returns resized face frames."""
    frames = []
    all_png = sorted(glob.glob(os.path.join(video_file, '*.png')))
    
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

    for png_path in all_png:
        img = cv2.imread(png_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(img)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                face_img = img[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (64, 64))
                frames.append(face_img)
        else:
            # If no face is detected, resize the entire frame
            img_resized = cv2.resize(img, (64, 64))
            frames.append(img_resized)
    
    return np.asarray(frames)


def read_wave(bvp_file):
    """Reads a BVP signal file."""
    with open(bvp_file, "r") as f:
        labels = json.load(f)
        waves = [label["Value"]["waveform"] for label in labels["/FullPackage"]]
    return np.asarray(waves)

def sample_data(data, num_points):
    """Samples `num_points` from `data` with a step size."""
    step_size = max(1, len(data) // num_points)
    return data[::step_size][:num_points]

def load_and_sample_PURE_data(base_path, num_points=1792):
    """Loads and samples video frames and BVP signals from the dataset."""
    data = {}

    for person_folder in os.listdir(base_path):
        print(f"Processing person {person_folder}")
        person_path = os.path.join(base_path, person_folder)
        if not os.path.isdir(person_path):
            continue

        videos = []
        bvps = []

        for session_folder in os.listdir(person_path):
            session_path = os.path.join(person_path, session_folder)
            if not os.path.isdir(session_path):
                continue

            #video_path = os.path.join(session_path, session_folder)
            video_frames = read_video(session_path)
            bvp_file = session_path + '.json'
            bvp_signal = read_wave(bvp_file)

            videos.append(video_frames)
            bvps.append(bvp_signal)

        if videos and bvps:
            combined_videos = np.concatenate(videos, axis=0)
            combined_bvps = np.concatenate(bvps, axis=0)

            sampled_videos = sample_data(combined_videos, num_points)
            sampled_bvps = sample_data(combined_bvps, num_points)

            data[person_folder] = {
                'videos': sampled_videos,
                'bvps': sampled_bvps
            }

    return data

def extract_PURE_videos_and_bvps(base_path):
    
    sampled_data = load_and_sample_PURE_data(base_path, num_points=1792)
    videos_pure = []
    bvps_pure = []

    for date in sampled_data:
        videos_pure.append(sampled_data[date]['videos'])
        bvps_pure.append(sampled_data[date]['bvps'])

    videos_pure = np.array(videos_pure)
    bvps_pure = np.array(bvps_pure)

    return videos_pure, bvps_pure



