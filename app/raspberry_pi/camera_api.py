from flask import Flask, jsonify, send_file, request
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from flask_cors import CORS
import os
import time
import threading
import subprocess
import cv2
import io
import numpy as np

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

BASE_DIR = os.path.expanduser('~/camera_files')
VIDEOS_DIR = os.path.join(BASE_DIR, 'videos')
os.makedirs(VIDEOS_DIR, exist_ok=True)

recording_state = {
    'is_recording': False,
    'current_file': None,
    'start_time': None,
    'duration': None
}

def convert_to_mov(input_file):
    """Convert H264 to MOV format"""
    try:
        output_file = input_file.replace('.h264', '.mov')
        print(f"Converting {input_file} to {output_file}")
        
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'h264',
            '-i', input_file,
            '-c:v', 'copy',  # Copy video stream without re-encoding
            '-f', 'mov',
            '-movflags', '+faststart',
            output_file
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
        video_config = picam2.create_video_configuration()
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
            recording_state.update({
                'is_recording': False,
                'current_file': os.path.basename(mov_file),
                'start_time': None,
                'duration': None
            })
            return os.path.basename(mov_file)
        else:
            print("Conversion failed")
            return filename
            
    except Exception as e:
        print(f"Recording error: {str(e)}")
        recording_state['is_recording'] = False
        return None
    
@app.route('/api/capture', methods=['POST', 'OPTIONS'])
def capture():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response, 204
        
    try:
        picam2 = Picamera2()
        capture_config = picam2.create_still_configuration(main={"size": (640, 480)})
        picam2.configure(capture_config)
        picam2.start()
        time.sleep(1)  # Let camera warm up
        
        image = picam2.capture_array()
        picam2.close()
        
        _, jpeg_data = cv2.imencode('.jpg', image)
        
        return send_file(
            io.BytesIO(jpeg_data.tobytes()),
            mimetype='image/jpeg'
        )
        
    except Exception as e:
        print(f"Capture error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/start', methods=['POST', 'OPTIONS'])
def start_video():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        if recording_state['is_recording']:
            return jsonify({'error': 'Already recording'}), 400
        
        data = request.get_json()
        duration = data.get('duration', 30)
        if duration > 300:
            return jsonify({'error': 'Duration too long'}), 400
            
        filename = f"video_{int(time.time())}.h264"
        
        recording_thread = threading.Thread(
            target=start_recording,
            args=(filename, duration)
        )
        recording_thread.start()
        
        recording_state.update({
            'is_recording': True,
            'current_file': filename.replace('.h264', '.mov'),
            'start_time': time.time(),
            'duration': duration
        })
        
        return jsonify({
            'status': 'recording',
            'filename': filename.replace('.h264', '.mov'),
            'duration': duration
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/download/<filename>', methods=['GET'])
def download_video(filename):
    try:
        file_path = os.path.join(VIDEOS_DIR, filename)
        print(f"Attempting to download: {file_path}")
        
        if os.path.exists(file_path):
            response = send_file(
                file_path,
                mimetype='video/quicktime',
                as_attachment=True,
                download_name=filename
            )
            response.headers['Accept-Ranges'] = 'bytes'
            return response
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Starting camera API on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=True)