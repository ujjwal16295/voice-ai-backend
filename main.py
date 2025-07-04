from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import tempfile
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import uuid
import librosa
import soundfile as sf
import numpy as np
import subprocess
import wave
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configure allowed file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'webm', 'mp4', 'ogg'}

# In-memory storage for processed audio files
audio_sessions = {}

# Load the BEST Demucs models for highest quality
print("Loading Demucs models...")
try:
    model = get_model('htdemucs_ft')
    print("Loaded htdemucs_ft (fine-tuned) - BEST QUALITY")
except:
    try:
        model = get_model('htdemucs_6s')
        print("Loaded htdemucs_6s (6-source) - HIGH QUALITY")
    except:
        model = get_model('htdemucs')
        print("Loaded htdemucs - STANDARD QUALITY")

print("Model loaded successfully!")

device = 'cpu'
model = model.to(device)
print(f"Using device: {device}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_with_ffmpeg_memory(input_bytes, input_format='mp3'):
    """Convert audio bytes using FFmpeg with high quality settings"""
    try:
        with tempfile.NamedTemporaryFile(suffix=f'.{input_format}', delete=False) as temp_input:
            temp_input.write(input_bytes)
            temp_input.flush()

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                cmd = [
                    'ffmpeg', '-i', temp_input.name,
                    '-ar', '44100',
                    '-ac', '2',
                    '-c:a', 'pcm_f32le',
                    '-y',
                    temp_output.name
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    with open(temp_output.name, 'rb') as f:
                        converted_data = f.read()

                    import os
                    os.unlink(temp_input.name)
                    os.unlink(temp_output.name)

                    return converted_data
                else:
                    print(f"FFmpeg error: {result.stderr}")
                    import os
                    os.unlink(temp_input.name)
                    os.unlink(temp_output.name)
                    return None

    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        return None


def fix_wav_bytes(input_bytes):
    """Try to fix corrupted WAV file by rewriting headers"""
    try:
        data_pos = input_bytes.find(b'data')
        if data_pos == -1:
            return None

        sample_rate = 44100
        channels = 1
        bits_per_sample = 16

        audio_data = input_bytes[data_pos + 8:]
        output_buffer = io.BytesIO()

        with wave.open(output_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(bits_per_sample // 8)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

        return output_buffer.getvalue()

    except Exception as e:
        print(f"WAV fix failed: {e}")
        return None


def enhance_vocal_clarity(vocals_tensor, sample_rate):
    """Enhanced vocal processing for maximum clarity"""
    try:
        audio_np = vocals_tensor.numpy()

        # Import scipy for advanced filtering
        try:
            from scipy import signal

            # 1. Vocal range emphasis (300Hz - 3kHz)
            # High-pass filter to remove low-frequency rumble
            sos_hp = signal.butter(4, 100, btype='high', fs=sample_rate, output='sos')

            # Band-pass emphasis for vocal clarity (boost vocal frequencies)
            sos_vocal = signal.butter(4, [300, 3000], btype='band', fs=sample_rate, output='sos')

            # 2. De-esser (reduce harsh sibilants)
            sos_deess = signal.butter(4, 8000, btype='low', fs=sample_rate, output='sos')

            # Apply filters to each channel
            if len(audio_np.shape) == 2:
                for i in range(audio_np.shape[0]):
                    # High-pass filter
                    audio_np[i] = signal.sosfilt(sos_hp, audio_np[i])

                    # Vocal range emphasis (gentle boost)
                    vocal_emphasis = signal.sosfilt(sos_vocal, audio_np[i])
                    audio_np[i] = audio_np[i] + 0.1 * vocal_emphasis

                    # Gentle de-essing
                    audio_np[i] = signal.sosfilt(sos_deess, audio_np[i])
            else:
                audio_np = signal.sosfilt(sos_hp, audio_np)
                vocal_emphasis = signal.sosfilt(sos_vocal, audio_np)
                audio_np = audio_np + 0.1 * vocal_emphasis
                audio_np = signal.sosfilt(sos_deess, audio_np)

            # 3. Dynamic range compression (gentle)
            def soft_compress(audio, threshold=0.7, ratio=3.0):
                """Gentle compression to even out vocal dynamics"""
                abs_audio = np.abs(audio)
                compressed = np.where(
                    abs_audio > threshold,
                    threshold + (abs_audio - threshold) / ratio,
                    abs_audio
                )
                return np.sign(audio) * compressed

            audio_np = soft_compress(audio_np)

            # 4. Noise gate (remove very quiet background noise)
            noise_gate_threshold = 0.005
            audio_np = np.where(np.abs(audio_np) < noise_gate_threshold, 0, audio_np)

        except ImportError:
            print("Scipy not available, applying basic vocal enhancement...")
            # Basic enhancement without scipy
            rms = np.sqrt(np.mean(audio_np ** 2))
            if rms > 0:
                audio_np = audio_np * (0.2 / rms)  # Normalize to reasonable level

        # 5. Final normalization with headroom
        max_val = np.max(np.abs(audio_np))
        if max_val > 0.85:
            audio_np = audio_np * (0.85 / max_val)

        return torch.from_numpy(audio_np.astype(np.float32))

    except Exception as e:
        print(f"Vocal enhancement failed: {e}")
        return vocals_tensor


def enhance_instrumental_clarity(instrumental_tensor, sample_rate):
    """Enhanced instrumental processing"""
    try:
        audio_np = instrumental_tensor.numpy()

        try:
            from scipy import signal

            # Remove vocal frequency bleed
            # Notch filter around vocal fundamental (reduce vocal remnants)
            sos_notch = signal.butter(2, [400, 2000], btype='bandstop', fs=sample_rate, output='sos')

            # Enhance bass and treble while reducing mids
            sos_bass = signal.butter(4, 200, btype='low', fs=sample_rate, output='sos')
            sos_treble = signal.butter(4, 4000, btype='high', fs=sample_rate, output='sos')

            if len(audio_np.shape) == 2:
                for i in range(audio_np.shape[0]):
                    # Light vocal frequency reduction
                    audio_np[i] = signal.sosfilt(sos_notch, audio_np[i]) * 0.95 + audio_np[i] * 0.05

                    # Enhance bass and treble
                    bass_enhanced = signal.sosfilt(sos_bass, audio_np[i])
                    treble_enhanced = signal.sosfilt(sos_treble, audio_np[i])
                    audio_np[i] = audio_np[i] + 0.05 * bass_enhanced + 0.05 * treble_enhanced
            else:
                audio_np = signal.sosfilt(sos_notch, audio_np) * 0.95 + audio_np * 0.05
                bass_enhanced = signal.sosfilt(sos_bass, audio_np)
                treble_enhanced = signal.sosfilt(sos_treble, audio_np)
                audio_np = audio_np + 0.05 * bass_enhanced + 0.05 * treble_enhanced

        except ImportError:
            print("Scipy not available, applying basic instrumental enhancement...")

        # Normalize
        max_val = np.max(np.abs(audio_np))
        if max_val > 0.85:
            audio_np = audio_np * (0.85 / max_val)

        return torch.from_numpy(audio_np.astype(np.float32))

    except Exception as e:
        print(f"Instrumental enhancement failed: {e}")
        return instrumental_tensor


def load_audio_from_bytes(file_bytes, filename):
    """Load audio from bytes with multiple fallback methods"""
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'mp3'
    print(f"Loading audio from bytes: {len(file_bytes)} bytes, format: {file_ext}")

    waveform = None
    sample_rate = None

    # Method 1: Try creating temporary file for librosa
    try:
        print("Trying librosa with temporary file...")
        with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_file.flush()

            audio, sample_rate = librosa.load(temp_file.name, sr=44100, mono=False, dtype=np.float32)

            import os
            os.unlink(temp_file.name)

            if len(audio.shape) == 1:
                audio = np.stack([audio, audio])
            elif len(audio.shape) == 2:
                if audio.shape[1] > audio.shape[0]:
                    audio = audio.T
                if audio.shape[0] == 1:
                    audio = np.vstack([audio, audio])
                elif audio.shape[0] > 2:
                    audio = audio[:2]

            waveform = torch.from_numpy(audio.astype(np.float32))
            print(f"Librosa success: shape={waveform.shape}, sr={sample_rate}")

    except Exception as librosa_error:
        print(f"Librosa failed: {librosa_error}")

        # Method 2: Try FFmpeg conversion
        converted_bytes = convert_with_ffmpeg_memory(file_bytes, file_ext)
        if converted_bytes:
            try:
                print("Trying FFmpeg converted bytes...")
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(converted_bytes)
                    temp_file.flush()

                    audio, sample_rate = librosa.load(temp_file.name, sr=44100, mono=False, dtype=np.float32)

                    import os
                    os.unlink(temp_file.name)

                    if len(audio.shape) == 1:
                        audio = np.stack([audio, audio])
                    waveform = torch.from_numpy(audio.astype(np.float32))
                    print(f"FFmpeg conversion success: shape={waveform.shape}")

            except Exception as e:
                print(f"FFmpeg converted file failed: {e}")

        # Method 3: Try WAV file fixing for WAV files
        if waveform is None and file_ext == 'wav':
            fixed_bytes = fix_wav_bytes(file_bytes)
            if fixed_bytes:
                try:
                    print("Trying fixed WAV bytes...")
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_file.write(fixed_bytes)
                        temp_file.flush()

                        audio, sample_rate = librosa.load(temp_file.name, sr=44100, mono=False, dtype=np.float32)

                        import os
                        os.unlink(temp_file.name)

                        if len(audio.shape) == 1:
                            audio = np.stack([audio, audio])
                        waveform = torch.from_numpy(audio.astype(np.float32))
                        print(f"Fixed WAV success: shape={waveform.shape}")

                except Exception as e:
                    print(f"Fixed WAV failed: {e}")

        # Method 4: Try torchaudio as last resort
        if waveform is None:
            try:
                print("Trying torchaudio...")
                with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as temp_file:
                    temp_file.write(file_bytes)
                    temp_file.flush()

                    waveform, sample_rate = torchaudio.load(temp_file.name)

                    import os
                    os.unlink(temp_file.name)

                    if waveform.shape[0] == 1:
                        waveform = waveform.repeat(2, 1)
                    elif waveform.shape[0] > 2:
                        waveform = waveform[:2]
                    print(f"Torchaudio success: shape={waveform.shape}")

            except Exception as torch_error:
                print(f"Torchaudio failed: {torch_error}")

    if waveform is None:
        raise Exception("Could not load audio file with any method")

    if sample_rate is None:
        sample_rate = 44100
        print("Using default sample rate: 44100")

    return waveform, sample_rate


def process_audio_memory(file_bytes, filename):
    """Process audio file with enhanced vocal clarity - all in memory"""
    try:
        print(f"Processing audio for maximum vocal clarity: {filename}")

        # Load audio from bytes
        waveform, sample_rate = load_audio_from_bytes(file_bytes, filename)

        # High-quality resampling if necessary
        if sample_rate != 44100:
            print(f"High-quality resampling from {sample_rate} to 44100...")
            resampler = torchaudio.transforms.Resample(
                sample_rate,
                44100,
                resampling_method='kaiser_window',
                rolloff=0.99,
                beta=14.769656459379492
            )
            waveform = resampler(waveform)
            sample_rate = 44100

        print(f"Final audio shape: {waveform.shape}, sample rate: {sample_rate}")

        # Apply Demucs model with optimized settings for clarity
        print("Applying Demucs model with clarity-focused settings...")

        # Use multiple shifts for better separation quality
        with torch.no_grad():
            sources = apply_model(
                model,
                waveform.unsqueeze(0),
                device=device,
                shifts=3,  # Multiple shifts for better quality
                split=True,
                overlap=0.25,  # Higher overlap for smoother results
                progress=False
            )[0]

        # Extract and enhance separated sources
        if sources.shape[0] == 6:  # 6-source model
            drums = sources[0]
            bass = sources[1]
            other = sources[2]
            vocals = sources[3]
            guitar = sources[4]
            piano = sources[5]
            instrumental = drums + bass + other + guitar + piano
        else:  # 4-source model
            drums = sources[0]
            bass = sources[1]
            other = sources[2]
            vocals = sources[3]
            instrumental = drums + bass + other

        # Apply specialized enhancement for vocal clarity
        print("Applying vocal clarity enhancement...")
        vocals_clean = enhance_vocal_clarity(vocals, sample_rate)

        print("Applying instrumental clarity enhancement...")
        instrumental_clean = enhance_instrumental_clarity(instrumental, sample_rate)

        # Convert to bytes for storage
        def tensor_to_bytes(tensor, sample_rate):
            buffer = io.BytesIO()
            audio_np = tensor.numpy().T
            sf.write(buffer, audio_np, sample_rate, format='WAV', subtype='PCM_16')
            buffer.seek(0)
            return buffer.getvalue()

        vocals_bytes = tensor_to_bytes(vocals_clean, sample_rate)
        instrumental_bytes = tensor_to_bytes(instrumental_clean, sample_rate)

        print("Ultra-clean vocal separation complete!")

        return {
            'vocals': vocals_bytes,
            'instrumental': instrumental_bytes
        }

    except Exception as e:
        print(f"Processing error: {str(e)}")
        raise Exception(f"Error processing audio: {str(e)}")


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Enhanced vocal clarity server running',
        'device': device,
        'model': model.__class__.__name__,
        'active_sessions': len(audio_sessions),
        'features': ['vocal_enhancement', 'noise_reduction', 'clarity_boost']
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Supported: mp3, wav, flac, m4a'}), 400

        session_id = str(uuid.uuid4())
        file_bytes = file.read()
        filename = secure_filename(file.filename)

        print(f"Received file: {filename}, size: {len(file_bytes)} bytes")

        # Process audio with enhanced clarity
        result_audio = process_audio_memory(file_bytes, filename)

        # Store results in memory
        audio_sessions[session_id] = {
            'vocals': result_audio['vocals'],
            'instrumental': result_audio['instrumental'],
            'filename': filename
        }

        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Audio processed with enhanced vocal clarity and noise reduction'
        })

    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/download/<session_id>/<file_type>', methods=['GET'])
def download_file(session_id, file_type):
    try:
        if file_type not in ['vocals', 'instrumental']:
            return jsonify({'error': 'Invalid file type'}), 400

        if session_id not in audio_sessions:
            return jsonify({'error': 'Session not found'}), 404

        session_data = audio_sessions[session_id]

        if file_type not in session_data:
            return jsonify({'error': 'File not found'}), 404

        audio_bytes = session_data[file_type]
        audio_buffer = io.BytesIO(audio_bytes)

        original_name = session_data['filename'].rsplit('.', 1)[0] if '.' in session_data['filename'] else session_data[
            'filename']

        quality_suffix = "ultra_clean" if file_type == "vocals" else "enhanced"

        return send_file(
            audio_buffer,
            as_attachment=True,
            download_name=f"{original_name}_{file_type}_{quality_suffix}.wav",
            mimetype='audio/wav'
        )

    except Exception as e:
        print(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/cleanup/<session_id>', methods=['DELETE'])
def cleanup_session(session_id):
    try:
        if session_id in audio_sessions:
            del audio_sessions[session_id]
            return jsonify({'success': True, 'message': 'Session cleaned up from memory'})
        else:
            return jsonify({'error': 'Session not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    return jsonify({
        'active_sessions': list(audio_sessions.keys()),
        'total_sessions': len(audio_sessions)
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
