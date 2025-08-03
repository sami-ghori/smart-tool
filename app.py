import os
import cv2
import gradio as gr
import numpy as np
import tempfile
import zipfile
import shutil
import webrtcvad
import soundfile as sf
import subprocess
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import noisereduce as nr

# --- Global Configurations and Dependencies ---
# These are loaded once to avoid overhead
CASCADE_PATH = Path(__file__).with_name("haarcascade_frontalface_default.xml")
if not CASCADE_PATH.exists():
    raise FileNotFoundError("Missing haarcascade_frontalface_default.xml!")

face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
vad = webrtcvad.Vad(2)

# --- Parameters (can be exposed to user via Gradio UI for more control) ---
FRAME_SKIP = 5
MIN_SEGMENT_DURATION = 25  # seconds
CROP_PADDING = 50          # pixels to add around the detected face

# --- Utility Functions ---

def run_ffmpeg_command(command, desc="FFmpeg task"):
    """Runs an FFmpeg command with error checking."""
    process = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"{desc} failed with error:\n{process.stderr}")

def extract_audio(video_path, audio_path):
    """Extracts mono audio at 16kHz from a video file."""
    command = ['ffmpeg', '-y', '-i', video_path, '-ar', '16000', '-ac', '1', audio_path]
    run_ffmpeg_command(command, "Audio extraction")

def isolate_speaker_voice(audio_path, cleaned_audio_path, sample_rate=16000):
    """
    Isolates speaker voice from noise using noisereduce (v3.x API).
    """
    try:
        audio, sr = sf.read(audio_path)
        if sr != sample_rate:
            pass

        # Correct API usage for noisereduce v3.x
        noise_profile = audio[:sample_rate]
        nr_instance = nr.NoiseReduce(y=noise_profile, sr=sample_rate)
        reduced_noise = nr_instance.reduce_noise(y=audio, sr=sample_rate)

        sf.write(cleaned_audio_path, reduced_noise, sample_rate)
    except Exception as e:
        raise RuntimeError(f"Noise reduction failed: {e}")

def get_speech_segments(audio_path, sample_rate=16000, window_ms=30):
    """Detects speech segments using WebRTC VAD."""
    audio, sr = sf.read(audio_path)
    pcm = (audio * 32768).astype(np.int16).tobytes()
    samples_per_window = int(sample_rate * window_ms / 1000)
    bytes_per_sample = 2
    segments = []

    for i in range(0, len(pcm), samples_per_window * bytes_per_sample):
        window = pcm[i:i + samples_per_window * bytes_per_sample]
        if len(window) < samples_per_window * bytes_per_sample:
            break
        if vad.is_speech(window, sample_rate):
            segments.append(i / (bytes_per_sample * sample_rate))
    return segments

def get_video_dims(video_path):
    """Gets the dimensions of a video file."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def detect_face_with_coords(video_path, progress=None):
    """
    Detects faces and records the bounding box coordinates.
    Returns a list of (timestamp, (x, y, w, h)).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    face_coords = []

    with tqdm(total=total_frames // FRAME_SKIP, desc="Detecting faces") as pbar:
        frame_idx = 0
        while True:
            ret = cap.grab()
            if not ret:
                break
            if frame_idx % FRAME_SKIP == 0:
                ret, frame = cap.retrieve()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                    face_coords.append((frame_idx / fps, (x, y, w, h)))

                pbar.update(1)
            frame_idx += 1

    cap.release()
    return face_coords

def merge_segments(face_coords, speech_times):
    """Merges face and speech segments."""
    face_set = set(int(t) for t, _ in face_coords)
    speech_set = set(int(t) for t in speech_times)
    merged_times = sorted(list(face_set & speech_set))

    if not merged_times: return []
    segments = []
    current_start = merged_times[0]
    for i in range(1, len(merged_times)):
        if merged_times[i] - merged_times[i-1] > 1.0:
            if merged_times[i-1] - current_start >= MIN_SEGMENT_DURATION:
                segments.append((current_start, merged_times[i-1]))
            current_start = merged_times[i]
    if merged_times[-1] - current_start >= MIN_SEGMENT_DURATION:
        segments.append((current_start, merged_times[-1]))
    return segments

def extract_and_crop_clips(video_path, segments, face_coords, output_dir, progress=None):
    """Extracts and re-encodes video clips with cropping."""
    video_width, video_height = get_video_dims(video_path)
    face_map = {int(t): coords for t, coords in face_coords}

    for i, (start, end) in enumerate(tqdm(segments, desc="Cropping and re-encoding clips")):
        output_path = os.path.join(output_dir, f"clip_{i + 1}.mp4")
        coords_in_segment = [face_map[t] for t in range(int(start), int(end) + 1) if t in face_map]

        if not coords_in_segment: continue
        min_x = min(c[0] for c in coords_in_segment)
        min_y = min(c[1] for c in coords_in_segment)
        max_x = max(c[0] + c[2] for c in coords_in_segment)
        max_y = max(c[1] + c[3] for c in coords_in_segment)

        x = max(0, min_x - CROP_PADDING)
        y = max(0, min_y - CROP_PADDING)
        w = min(video_width - x, max_x - min_x + 2 * CROP_PADDING)
        h = min(video_height - y, max_y - min_y + 2 * CROP_PADDING)
        crop_filter = f"crop={w}:{h}:{x}:{y}"

        command = [
            'ffmpeg', '-y', '-i', video_path, '-ss', str(start), '-to', str(end),
            '-vf', crop_filter, '-c:v', 'libx264', '-crf', '23', 
            '-c:a', 'aac', '-b:a', '128k', output_path
        ]
        run_ffmpeg_command(command, f"Clip {i+1} extraction")
        if progress: progress((i + 1) / len(segments))

def process_video_pipeline(video_file, progress=gr.Progress(track_tqdm=True)):
    """Main function to orchestrate the entire video processing pipeline."""
    temp_dir = tempfile.mkdtemp()
    try:
        video_path = shutil.copy(video_file.name, os.path.join(temp_dir, "input_video.mp4"))
        audio_path = os.path.join(temp_dir, "audio.wav")
        cleaned_audio_path = os.path.join(temp_dir, "cleaned_audio.wav")

        progress(0.05, desc="Starting parallel processing...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_audio = executor.submit(process_audio_pipeline, video_path, audio_path, cleaned_audio_path)
            future_faces = executor.submit(detect_face_with_coords, video_path)

            progress(0.1, desc="Processing audio and video in parallel...")
            speech_times = future_audio.result()
            face_coords = future_faces.result()

        progress(0.7, desc="Merging segments and preparing for clipping...")
        segments = merge_segments(face_coords, speech_times)
        if not segments: raise gr.Error("⚠️ No valid face + voice segments detected.")

        clips_dir = os.path.join(temp_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        progress(0.8, desc="Cropping and re-encoding video clips...")
        extract_and_crop_clips(video_path, segments, face_coords, clips_dir, progress=progress)

        zip_path = os.path.join(temp_dir, "clips.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for clip in os.listdir(clips_dir): zipf.write(os.path.join(clips_dir, clip), arcname=clip)
        if not os.path.exists(zip_path): raise gr.Error("Unexpected error: ZIP file not created.")
        return zip_path
    except gr.Error as ge:
        raise ge
    except Exception as e:
        print(f"Error: {e}")
        raise gr.Error(f"❌ Internal error during processing: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def process_audio_pipeline(video_path, audio_path, cleaned_audio_path):
    """Sub-pipeline for all audio-related tasks."""
    extract_audio(video_path, audio_path)
    isolate_speaker_voice(audio_path, cleaned_audio_path)
    return get_speech_segments(cleaned_audio_path)

with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Smart Speaker Extractor with Face + Voice Detection")
    gr.Markdown("This tool will extract video clips of a single speaker, cropping the video to focus only on their face and isolating their voice from background noise.")
    video_input = gr.File(label="Upload a video", type="filepath")
    output_file = gr.File(label="Download ZIP of segments")
    submit_btn = gr.Button("Start Processing", variant="primary")
    submit_btn.click(fn=process_video_pipeline, inputs=[video_input], outputs=[output_file])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)