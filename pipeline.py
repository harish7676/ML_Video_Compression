import cv2
import numpy as np
import pickle
import zlib
import os
import math
import sys
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

class VideoCompressor:
    def __init__(self, keyframe_interval=10, quality=25):
        # Parameters for feature detection and tracking
        self.feature_params = dict(
            maxCorners=1000,  # Maximum number of corners to track
            qualityLevel=0.3,  # Minimum quality of corner points
            minDistance=7,     # Minimum distance between tracked points
            blockSize=7
        )
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),  # Search window size for tracking
            maxLevel=2,        # Pyramid levels for multiscale tracking
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Tracking termination criteria
        )
        self.keyframe_interval = keyframe_interval  # Interval between keyframes
        self.quality = quality  # JPEG compression quality (lower means higher compression)

    def simple_motion_predictor(self, flow_data):
        """
        Simple motion prediction using Linear Regression
        Predicts movement of points based on previous frames
        """
        if not flow_data:
            return None

        # Prepare data for regression
        X, y_x, y_y = [], [], []
        for old_points, new_points in flow_data:
            for old, new in zip(old_points, new_points):
                X.append([old[0], old[1]])  # Original point coordinates
                y_x.append(new[0] - old[0])  # X-axis movement
                y_y.append(new[1] - old[1])  # Y-axis movement

        # Train linear regression models for X and Y movement
        if len(X) > 0:
            reg_x = LinearRegression().fit(X, y_x)
            reg_y = LinearRegression().fit(X, y_y)
            return (reg_x, reg_y)

        return None

    def compress_video(self, video_path):
        """
        Compress video by extracting keyframes and tracking optical flow
        """
        cap = cv2.VideoCapture(video_path)
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        keyframes, flow_data = [], []
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read the first frame of the video.")

        # Convert first frame to grayscale for feature tracking
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_idx = 0

        # Add progress bar for frame processing
        pbar = tqdm(total=total_frames, desc="Compressing Video", unit="frames")

        while True:
            ret, next_frame = cap.read()
            if not ret:
                break

            # Convert next frame to grayscale
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect good features to track
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
            
            if p0 is not None:
                # Track features using optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **self.lk_params)

                if st is not None:
                    # Store valid tracked points
                    good_old = p0[st.flatten() == 1]
                    good_new = p1[st.flatten() == 1]
                    flow_segment = ([point.flatten().tolist() for point in good_old],
                                    [point.flatten().tolist() for point in good_new])
                    flow_data.append(flow_segment)

            # Store keyframes at specified intervals
            if frame_idx % self.keyframe_interval == 0:
                ret, compressed_keyframe = cv2.imencode('.jpg', prev_frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if ret:
                    keyframes.append(compressed_keyframe.tobytes())

            prev_gray = next_gray.copy()
            prev_frame = next_frame.copy()
            frame_idx += 1
            pbar.update(1)

        pbar.close()

        # Compute motion prediction model
        ml_motion_model = self.simple_motion_predictor(flow_data)
        
        # Add last frame as keyframe
        ret, compressed_keyframe = cv2.imencode('.jpg', prev_frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ret:
            keyframes.append(compressed_keyframe.tobytes())

        cap.release()

        # Compress flow data for storage
        flow_data_compressed = zlib.compress(pickle.dumps(flow_data))

        # Store metadata about the compression process
        metadata = {
            "original_fps": original_fps,
            "total_frames": total_frames,
            "keyframe_interval": self.keyframe_interval
        }

        # Prepare compressed data for saving
        compressed_data = {
            "keyframes": keyframes,
            "flow_data": flow_data_compressed,
            "ml_motion_model": ml_motion_model,
            "metadata": metadata
        }

        return compressed_data

    def reconstruct_video(self, compressed_data, output_path):
        """
        Reconstruct video from compressed keyframes and motion data
        """
        keyframes = compressed_data["keyframes"]
        flow_data = pickle.loads(zlib.decompress(compressed_data["flow_data"]))
        ml_motion_model = compressed_data.get("ml_motion_model")
        metadata = compressed_data["metadata"]

        # Retrieve original video parameters
        original_fps = metadata["original_fps"]
        total_frames = metadata["total_frames"]
        keyframe_interval = metadata["keyframe_interval"]

        # Get frame dimensions from first keyframe
        first_frame = cv2.imdecode(np.frombuffer(keyframes[0], np.uint8), cv2.IMREAD_COLOR)
        height, width = first_frame.shape[:2]

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))

        # Add progress bar for reconstruction
        pbar = tqdm(total=len(keyframes), desc="Reconstructing Video", unit="keyframe")

        # Reconstruct video from keyframes
        for i in range(len(keyframes) - 1):
            current_keyframe = cv2.imdecode(np.frombuffer(keyframes[i], np.uint8), cv2.IMREAD_COLOR)
            next_keyframe = cv2.imdecode(np.frombuffer(keyframes[i + 1], np.uint8), cv2.IMREAD_COLOR)

            # Write current keyframe
            out.write(current_keyframe)

            # Interpolate intermediate frames if motion model exists
            if ml_motion_model:
                for j in range(1, keyframe_interval):
                    t = j / keyframe_interval
                    intermediate_frame = self.interpolate_frame(current_keyframe, next_keyframe, ml_motion_model, t)
                    out.write(intermediate_frame)

            pbar.update(1)

        pbar.close()
        out.release()
        return output_path

    def interpolate_frame(self, frame1, frame2, model, t):
        """
        Simple linear frame interpolation
        """
        return cv2.addWeighted(frame1, 1 - t, frame2, t, 0)

class VideoQualityAssessment:
    def __init__(self, original_video_path, reconstructed_video_path):
        """
        Initialize video quality assessment with original and reconstructed video paths
        """
        self.original_video_path = original_video_path
        self.reconstructed_video_path = reconstructed_video_path

    def calculate_mae(self, frame1, frame2):
        """
        Calculate Mean Absolute Error between two frames
        Measures average pixel-wise difference
        """
        return np.mean(np.abs(frame1.astype(np.float32) - frame2.astype(np.float32)))

    def calculate_psnr(self, frame1, frame2):
        """
        Calculate Peak Signal-to-Noise Ratio
        Higher values indicate better quality (lower distortion)
        """
        mse = np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2)
        if mse == 0:
            return 100
        max_pixel = 255.0
        return 10 * math.log10((max_pixel ** 2) / mse)

    def assess_video_quality(self):
        """
        Compare original and reconstructed videos
        Calculate MAE and PSNR for each frame
        """
        cap_orig = cv2.VideoCapture(self.original_video_path)
        cap_reconstructed = cv2.VideoCapture(self.reconstructed_video_path)

        # Get total frame count for progress bar
        total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))

        mae_values, psnr_values = [], []

        # Add progress bar for quality assessment
        pbar = tqdm(total=total_frames, desc="Assessing Video Quality", unit="frames")

        while True:
            ret_orig, orig_frame = cap_orig.read()
            ret_reconstructed, reconstructed_frame = cap_reconstructed.read()

            # Stop when any video ends
            if not ret_orig or not ret_reconstructed:
                break

            # Resize frames to match dimensions
            if orig_frame.shape != reconstructed_frame.shape:
                reconstructed_frame = cv2.resize(reconstructed_frame, (orig_frame.shape[1], orig_frame.shape[0]))

            # Calculate quality metrics
            mae = self.calculate_mae(orig_frame, reconstructed_frame)
            psnr = self.calculate_psnr(orig_frame, reconstructed_frame)

            mae_values.append(mae)
            psnr_values.append(psnr)

            pbar.update(1)

        pbar.close()
        cap_orig.release()
        cap_reconstructed.release()

        # Return average metrics
        return {
            'average_mae': np.mean(mae_values),
            'average_psnr': np.mean(psnr_values)
        }

def process_video(video_file_path):
    """
    Main processing function to compress, reconstruct, and assess video quality
    """
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Define paths for compressed and reconstructed videos
    compressed_video_path = 'output/compressed_video_data.pkl'
    reconstructed_video_path = 'output/reconstructed_video.avi'

    # Initialize video compressor
    compressor = VideoCompressor()

    # Compress video and save compressed data
    print("Compressing video...")
    compressed_data = compressor.compress_video(video_file_path)
    with open(compressed_video_path, 'wb') as f:
        pickle.dump(compressed_data, f)

    # Reconstruct video from compressed data
    print("Reconstructing video...")
    compressor.reconstruct_video(compressed_data, reconstructed_video_path)

    # Assess video quality
    print("Assessing video quality...")
    quality_assessor = VideoQualityAssessment(video_file_path, reconstructed_video_path)
    metrics = quality_assessor.assess_video_quality()

    # Print quality metrics
    print("\nVideo Compression Performance Metrics:")
    print(f"Mean Absolute Error (MAE): {metrics['average_mae']:.4f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {metrics['average_psnr']:.4f} dB")

    return metrics

if __name__ == "__main__":
    # Check if a path is provided as a command-line argument
    input_video = sys.argv[1] if len(sys.argv) > 1 else "input_video.mp4"
    process_video(input_video)