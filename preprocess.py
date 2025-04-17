import cv2
import os

def extract_frames(video_name, output, start, end, FPS=0.0005, human=True):
    human_vid = "./data/Human Lymphatics-selected/"
    rat_vid = "./data/Rat Lymphatics"
    video_path = human_vid if human else rat_vid

    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    full_path = None

    for ext in video_extensions:
        trial_path = os.path.join(video_path, video_name + ext)
        if os.path.exists(trial_path):
            full_path = trial_path
            break

    if full_path is None:
        print(f"Video file not found for {video_name} in: {video_path}")
        return

    cap = cv2.VideoCapture(full_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    start = max(0, start)
    end = min(duration, end)

    print(f"Video: {video_name}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total}")
    print(f"Video duration: {duration:.2f} seconds")
    print(f"Extracting from {start:.2f}s to {end:.2f}s")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Resolution: {width}x{height}")

    if not os.path.exists(output):
        os.makedirs(output)

    interval = int(fps / FPS)
    print("Interval: ",interval)
    idx = int(start * fps)
    max_frame = int(end * fps)
    saved = 0

    while idx <= max_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = idx / fps
        minutes = int(time_sec // 60)
        seconds = int(time_sec % 60)
        milliseconds = int((time_sec - int(time_sec)) * 1000)

        frame_name = f"{minutes:02d}_{seconds:02d}_{milliseconds:03d}.png"
        frame_path = os.path.join(output, frame_name)
        cv2.imwrite(frame_path, frame)

        saved += 1
        idx += interval

    cap.release()
    print(f"✅ Done: {saved} frames saved.")

vid_name = "Human_Lymphatic_02-12-24_pressure_0ca_scan_East2"
output = "./output/human/" + vid_name
# extract_frames(vid_name, output, 2, 8,1,False)
extract_frames(vid_name, output, 157, 167 ,1)


