from flask import Flask, jsonify, send_file, request
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from flask_cors import CORS
from threading import Lock
import os
import time
import threading
import subprocess
import cv2
import io
import numpy as np
import threading


# Updated version that doesn't require the livepeer package
class LivepeerStreamService:
    camera_lock = Lock()  # Class-level lock for camera access

    def _acquire_camera(self):
        """Try to get camera access, stop stream if needed"""
        if self.is_streaming:
            self.stop_streaming()
            time.sleep(1)  # Give camera time to release
        return self.camera_lock.acquire()

    def _release_camera(self):
        self.camera_lock.release()

    def __init__(self):
        self.api_key = os.getenv("LIVEPEER_API_KEY")
        self.stream_key = os.getenv("LIVEPEER_STREAM_KEY")
        self.ingest_url = os.getenv("LIVEPEER_INGEST_URL")
        self.streaming_process = None
        self.is_streaming = False

    def start_streaming(self):
        """Start streaming from camera using Picamera2"""
        if not self.stream_key or not self.ingest_url:
            print("Missing stream key or ingest URL")
            return False

        if self.is_streaming:
            print("Already streaming")
            return False

        try:
            rtmp_url = f"{self.ingest_url}/{self.stream_key}"
            print(f"Attempting to stream to: {rtmp_url}")

            picam2 = Picamera2()
            video_config = picam2.create_video_configuration(
                main={"size": (1280, 720), "format": "RGB888"},
                controls={
                    "FrameRate": 30.0,
                    "ExposureTime": 150000,  # Increased more for brightness
                    "AnalogueGain": 5.0,  # Increased gain further
                    "Saturation": 1.2,  # Increased from 0.5 to boost color
                    "Brightness": 0.2,  # Increased brightness
                    "Contrast": 1.0,  # Set to neutral to avoid color distortion
                    "Sharpness": 1.5,  # Added to improve clarity
                },
            )

            picam2.configure(video_config)
            picam2.start()

            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",  # This line changes from bgr24 to rgb24
                "-s",
                "1280x720",
                "-r",
                "30",
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "ultrafast",
                "-f",
                "flv",
                rtmp_url,
            ]

            self.streaming_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            def stream_frames():
                try:
                    while self.is_streaming:
                        frame = picam2.capture_array()
                        self.streaming_process.stdin.write(frame.tobytes())
                finally:
                    picam2.stop()
                    if self.streaming_process:
                        self.streaming_process.terminate()

            self.is_streaming = True

            stream_thread = threading.Thread(target=stream_frames)
            stream_thread.daemon = True
            stream_thread.start()

            return True

        except Exception as e:
            print(f"Error starting stream: {e}")
            return False

    def stop_streaming(self):
        if self.streaming_process:
            try:
                # Kill the entire process group (libcamera-vid and ffmpeg)
                cmd = f"pkill -f 'libcamera-vid|ffmpeg'"
                subprocess.run(cmd, shell=True)
                self.streaming_process = None
                self.is_streaming = False
                print("Stream stopped")
            except Exception as e:
                print(f"Error stopping stream: {e}")

    def get_stream_info(self):
        return {
            "isActive": self.is_streaming,
            "ingestUrl": self.ingest_url if self.ingest_url else None,
            "streamKey": self.stream_key if self.stream_key else None,
            "playbackId": "a1831mk3ncwwk8cu",
        }


app = Flask(__name__)
CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "Accept"],
        }
    },
)
stream_service = LivepeerStreamService()


# Update the endpoint to use the new service
@app.route("/api/stream/info", methods=["GET"])
def get_stream_info():
    """Get current stream information"""
    try:
        info = stream_service.get_stream_info()
        if info:
            return jsonify(info)
        return jsonify({"error": "No stream info available"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stream/health", methods=["GET"])
def check_stream_health():
    """Check if streaming service is available"""
    return jsonify(
        {
            "streaming_enabled": bool(stream_service.api_key),
            "has_active_stream": stream_service.is_streaming,
        }
    )


@app.route("/api/stream/start", methods=["POST"])
def start_stream():
    """Start streaming from camera"""
    try:
        success = stream_service.start_streaming()
        if success:
            return jsonify(stream_service.get_stream_info())
        return jsonify({"error": "Failed to start streaming"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stream/stop", methods=["POST"])
def stop_stream():
    """Stop the current stream"""
    try:
        stream_service.stop_streaming()
        return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


BASE_DIR = os.path.expanduser("~/camera_files")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

recording_state = {
    "is_recording": False,
    "current_file": None,
    "start_time": None,
    "duration": None,
}


def convert_to_mov(input_file):
    """Convert H264 to MOV format"""
    try:
        output_file = input_file.replace(".h264", ".mov")
        print(f"Converting {input_file} to {output_file}")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "h264",
            "-i",
            input_file,
            "-c:v",
            "copy",  # Copy video stream without re-encoding
            "-f",
            "mov",
            "-movflags",
            "+faststart",
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return None

        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"Successfully converted to MOV: {output_file}")
            print(f"File size: {os.path.getsize(output_file)} bytes")
            return output_file
        else:
            print("Conversion failed: Output file is empty or doesn't exist")
            return None

    except Exception as e:
        print(f"Conversion error: {str(e)}")
        return None


def start_recording(filename, duration):
    try:
        print(f"Starting recording for {duration} seconds to {filename}")
        picam2 = Picamera2()
        video_config = picam2.create_video_configuration(
            main={"size": (1280, 720)},
            controls={
                "FrameRate": 30.0,
                "ExposureTime": 75000,  # Increased from default
                "AnalogueGain": 2.0,  # Increased from default
            },
        )
        picam2.configure(video_config)
        encoder = H264Encoder(bitrate=10000000)

        h264_file = os.path.join(VIDEOS_DIR, filename)
        print(f"Recording to H264 file: {h264_file}")

        picam2.start_recording(encoder, h264_file)
        time.sleep(duration)
        picam2.stop_recording()
        picam2.close()

        print("Converting to MOV...")
        mov_file = convert_to_mov(h264_file)

        if mov_file:
            print(f"Conversion successful: {mov_file}")
            os.remove(h264_file)
            recording_state.update(
                {
                    "is_recording": False,
                    "current_file": os.path.basename(mov_file),
                    "start_time": None,
                    "duration": None,
                }
            )
            return os.path.basename(mov_file)
        else:
            print("Conversion failed")
            return filename

    except Exception as e:
        print(f"Recording error: {str(e)}")
        recording_state["is_recording"] = False
        return None


@app.route('/api/capture', methods=['POST', 'OPTIONS'])
def capture():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response, 204

    was_streaming = stream_service.is_streaming
    try:
        # If streaming, stop it temporarily
        if was_streaming:
            stream_service.stop_streaming()
            time.sleep(1)  # Give camera time to release
            
        picam2 = Picamera2()
        capture_config = picam2.create_still_configuration(
            main={"size": (1280, 720), "format": "RGB888"},
            controls={
                "ExposureTime": 150000,
                "AnalogueGain": 5.0,
                "Saturation": 1.2,
                "Brightness": 0.2,
                "Contrast": 1.0,
                "Sharpness": 1.5
            }
        )
        picam2.configure(capture_config)
        picam2.start()
        time.sleep(1)  # Let camera warm up
        
        image = picam2.capture_array()
        picam2.close()
        
        _, jpeg_data = cv2.imencode('.jpg', image)
        
        # If we were streaming before, restart it
        if was_streaming:
            stream_service.start_streaming()
        
        return send_file(
            io.BytesIO(jpeg_data.tobytes()),
            mimetype='image/jpeg'
        )
        
    except Exception as e:
        print(f"Capture error: {e}")
        # Try to restart streaming if it was active
        if was_streaming:
            stream_service.start_streaming()
        return jsonify({'error': str(e)}), 500


@app.route("/api/video/start", methods=["POST", "OPTIONS"])
def start_video():
    if request.method == "OPTIONS":
        return "", 204

    try:
        if recording_state["is_recording"]:
            return jsonify({"error": "Already recording"}), 400

        data = request.get_json()
        duration = data.get("duration", 30)
        if duration > 300:
            return jsonify({"error": "Duration too long"}), 400

        filename = f"video_{int(time.time())}.h264"

        recording_thread = threading.Thread(
            target=start_recording, args=(filename, duration)
        )
        recording_thread.start()

        recording_state.update(
            {
                "is_recording": True,
                "current_file": filename.replace(".h264", ".mov"),
                "start_time": time.time(),
                "duration": duration,
            }
        )

        return jsonify(
            {
                "status": "recording",
                "filename": filename.replace(".h264", ".mov"),
                "duration": duration,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/download/<filename>", methods=["GET"])
def download_video(filename):
    try:
        file_path = os.path.join(VIDEOS_DIR, filename)
        print(f"Attempting to download: {file_path}")

        if os.path.exists(file_path):
            response = send_file(
                file_path,
                mimetype="video/quicktime",
                as_attachment=True,
                download_name=filename,
            )
            response.headers["Accept-Ranges"] = "bytes"
            return response
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Starting camera API on port 5001...")
    app.run(host="0.0.0.0", port=5001, debug=True)
