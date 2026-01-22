# Copyright (c) ModelScope Contributors. All rights reserved.
"""
VSI-Bench Frame Extraction Script

Extract frames from videos in advance to speed up evaluation.
This avoids repeated video decoding during inference.

Usage:
    python extract_frames.py \
        --video_dir /path/to/VSI-Bench \
        --output_dir /path/to/VSI-Bench-frames \
        --num_frames 32 \
        --num_workers 8
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    num_frames: int = 32,
) -> Tuple[str, bool, str]:
    """Extract uniformly sampled frames from a video.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract

    Returns:
        Tuple of (video_path, success, message)
    """
    try:
        # Try decord first (faster)
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)

            # Uniformly sample frame indices
            if total_frames <= num_frames:
                indices = list(range(total_frames))
            else:
                indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Extract and save frames
            for i, idx in enumerate(indices):
                frame = vr[idx].asnumpy()
                # Convert RGB to PIL Image and save
                from PIL import Image
                img = Image.fromarray(frame)
                frame_path = os.path.join(output_dir, f'frame_{i:04d}.jpg')
                img.save(frame_path, 'JPEG', quality=95)

            return video_path, True, f'Extracted {len(indices)} frames'

        except ImportError:
            # Fallback to cv2
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                cap.release()
                return video_path, False, 'Could not read video frames'

            # Uniformly sample frame indices
            if total_frames <= num_frames:
                indices = list(range(total_frames))
            else:
                indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Extract and save frames
            for i, idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_path = os.path.join(output_dir, f'frame_{i:04d}.jpg')
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            cap.release()
            return video_path, True, f'Extracted {len(indices)} frames (cv2)'

    except Exception as e:
        return video_path, False, str(e)


def find_videos(video_dir: str) -> List[str]:
    """Find all video files in directory recursively.

    Args:
        video_dir: Root directory to search

    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    videos = []

    for root, _, files in os.walk(video_dir):
        for file in files:
            if Path(file).suffix.lower() in video_extensions:
                videos.append(os.path.join(root, file))

    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(description='Extract frames from VSI-Bench videos')
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for frames')
    parser.add_argument('--num_frames', type=int, default=32, help='Number of frames to extract per video')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (default: min(16, cpu_count))')
    parser.add_argument('--skip_existing', action='store_true', help='Skip videos with existing frames')
    args = parser.parse_args()

    # Auto-detect number of CPUs if not specified, with reasonable limit
    if args.num_workers is None:
        cpu_count = os.cpu_count() or 8
        args.num_workers = min(16, cpu_count)  # Cap at 16 to avoid resource issues
        print(f'Using {args.num_workers} workers (auto-detected, cpu_count={cpu_count})')

    # Find all videos
    print(f'Searching for videos in {args.video_dir}...')
    videos = find_videos(args.video_dir)
    print(f'Found {len(videos)} videos')

    if not videos:
        print('No videos found!')
        return

    # Prepare tasks
    tasks = []
    for video_path in videos:
        # Compute relative path to maintain directory structure
        rel_path = os.path.relpath(video_path, args.video_dir)
        # Remove extension for output directory name
        rel_path_no_ext = os.path.splitext(rel_path)[0]
        output_subdir = os.path.join(args.output_dir, rel_path_no_ext)

        # Skip if frames already exist
        if args.skip_existing and os.path.isdir(output_subdir):
            existing_frames = [f for f in os.listdir(output_subdir) if f.endswith('.jpg')]
            if len(existing_frames) >= args.num_frames:
                continue

        tasks.append((video_path, output_subdir, args.num_frames))

    print(f'Processing {len(tasks)} videos (skipped {len(videos) - len(tasks)} existing)...')

    # Process videos in parallel
    success_count = 0
    fail_count = 0
    failed_videos = []

    if args.num_workers == 1:
        # Sequential processing
        for video, output, num_frames in tqdm(tasks, desc='Extracting frames'):
            video_path, success, message = extract_frames_from_video(video, output, num_frames)
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_videos.append((video_path, message))
    else:
        # Parallel processing
        try:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {
                    executor.submit(extract_frames_from_video, video, output, args.num_frames): video
                    for video, output, _ in tasks
                }

                for future in tqdm(as_completed(futures), total=len(futures), desc='Extracting frames'):
                    try:
                        video_path, success, message = future.result()
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                            failed_videos.append((video_path, message))
                    except Exception as e:
                        fail_count += 1
                        failed_videos.append((futures[future], str(e)))

        except OSError as e:
            if 'Resource temporarily unavailable' in str(e) or 'Cannot allocate memory' in str(e):
                print(f'\nError: Too many parallel processes (num_workers={args.num_workers})')
                print('Try: --num_workers 8  or  --num_workers 4  or  --num_workers 1')
                return
            raise

    # Print summary
    print(f'\nExtraction complete!')
    print(f'  Success: {success_count}')
    print(f'  Failed: {fail_count}')

    if failed_videos:
        print(f'\nFailed videos:')
        for video, msg in failed_videos[:10]:
            print(f'  {video}: {msg}')
        if len(failed_videos) > 10:
            print(f'  ... and {len(failed_videos) - 10} more')


if __name__ == '__main__':
    main()
