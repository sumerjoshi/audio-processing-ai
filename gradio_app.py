import argparse
import os 
import tempfile
import zipfile
import shutil
import time
from pathlib import Path 
from datetime import datetime
import numpy as np
import traceback

# Conditional imports for Modal deploy-time parsing
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import matplotlib
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    matplotlib = None

# Conditional imports - only import torch if needed (for local PyTorch mode)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from predict import (
    preprocess_audio,
    predict_ai_only,
    get_audio_files,
    DualHeadCnn14Simple,
    analyze_music_simple,
    predict_folder
)

SAMPLE_RATE = 16000
DURATION = 10.0
AUDIO_LENGTH = int(SAMPLE_RATE * DURATION)
DEFAULT_THRESHOLD = 0.35

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
    
    # Monkey-patch to fix Gradio schema bug (bool instead of dict)
    try:
        from gradio_client import utils as gradio_client_utils
        # Patch get_type to handle boolean schemas
        original_get_type = gradio_client_utils.get_type
        def patched_get_type(schema):
            if isinstance(schema, bool):
                return "bool"
            if not isinstance(schema, dict):
                return str(type(schema).__name__)
            return original_get_type(schema)
        gradio_client_utils.get_type = patched_get_type
        
        # Patch _json_schema_to_python_type to handle boolean additionalProperties
        original_json_schema = gradio_client_utils._json_schema_to_python_type
        def patched_json_schema(schema, defs=None):
            if isinstance(schema, bool):
                return "bool"
            if isinstance(schema, dict) and 'additionalProperties' in schema:
                if isinstance(schema['additionalProperties'], bool):
                    # If additionalProperties is a boolean, treat it as allowing any type
                    return "dict"
            return original_json_schema(schema, defs)
        gradio_client_utils._json_schema_to_python_type = patched_json_schema
    except (ImportError, AttributeError):
        pass  # If patching fails, continue anyway
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

# Device definition - only needed for PyTorch mode
if TORCH_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = None  # Not needed for ONNX mode

class GradioAudioInterface:
    def __init__(self, model_path: str, threshold: float = DEFAULT_THRESHOLD, use_onnx: bool = False, is_modal: bool = False):
        self.model_path = model_path
        self.threshold = threshold
        self.onnx = use_onnx 
        self.is_modal = is_modal
        
        if use_onnx:
            if not ONNXRUNTIME_AVAILABLE:
                raise ImportError("onnxruntime is required for ONNX models. Install with: pip install onnxruntime")
            try:
                print(f"[{time.strftime('%H:%M:%S')}] Loading ONNX model from: {model_path}")
                
                # Configure ONNX session options for stability
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1  # Avoid threading issues in container
                sess_options.inter_op_num_threads = 1
                
                # Try to use GPU if available (for Modal GPU instances)
                # NOTE: Start with CPU to avoid GPU provider issues that can cause hangs
                providers = ['CPUExecutionProvider']
                
                if is_modal:
                    # On Modal with GPU, try CUDA provider but only if explicitly needed
                    # CPU is more reliable and avoids hanging issues
                    available_providers = ort.get_available_providers()
                    if 'CUDAExecutionProvider' in available_providers:
                        # Add CUDA as fallback (will use CPU first, then GPU if CPU fails)
                        # Actually, let's use CPU only for now to avoid hanging
                        print(f"[{time.strftime('%H:%M:%S')}] CUDAExecutionProvider available but using CPU for stability")
                        # providers.insert(0, 'CUDAExecutionProvider')  # Uncomment to try GPU
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] CUDAExecutionProvider not available, using CPU")
                
                print(f"[{time.strftime('%H:%M:%S')}] Using providers: {providers}")
                
                load_start = time.time()
                self.onnx_session = ort.InferenceSession(
                    model_path, 
                    sess_options,
                    providers=providers
                )
                load_time = time.time() - load_start
                print(f"[{time.strftime('%H:%M:%S')}] ONNX model loaded in {load_time:.2f}s")
                print(f"[{time.strftime('%H:%M:%S')}] Using providers: {self.onnx_session.get_providers()}")
                
                # Log model input/output info
                for input_info in self.onnx_session.get_inputs():
                    print(f"[{time.strftime('%H:%M:%S')}] Model input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
                for output_info in self.onnx_session.get_outputs():
                    print(f"[{time.strftime('%H:%M:%S')}] Model output: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
                
                self.model = None
            except Exception as e:
                error_msg = f"Failed to load ONNX model from {model_path}: {str(e)}"
                print(f"[{time.strftime('%H:%M:%S')}] {error_msg}")
                print(f"[{time.strftime('%H:%M:%S')}] Traceback: {traceback.format_exc()}")
                raise RuntimeError(error_msg)
        else:
            # PyTorch model loading
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for PyTorch models. Install with: pip install torch")
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = DualHeadCnn14Simple(pretrained=False)
            self.model.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=False))
            self.model.eval().to(device)
            self.onnx_session = None
        
        
    def __predict_single_audio(self, audio_file):
        """Predict if a single file is autogenerated"""
        empty_df = pd.DataFrame(columns=['Filename', 'AI-Generated', 'Confidence', 'Genre', 'Mood', 'Tempo (BPM)', 'Energy'])
        
        # Early return for None or empty values
        if audio_file is None:
            return "Please upload an audio file", None, empty_df 
        
        # Handle list inputs (Modal/Gradio might return lists)
        if isinstance(audio_file, list):
            if len(audio_file) == 0:
                return "Please upload an audio file", None, empty_df
            # Take first element if it's a list
            audio_file = audio_file[0]
        
        # Handle Gradio File component - it returns a file path string or a dict with 'name' key
        # In Modal, Gradio might handle file uploads differently
        if isinstance(audio_file, dict):
            # Gradio File component may return a dict with file info
            audio_file = audio_file.get('name', audio_file.get('path', None))
            if not audio_file:
                return "Error: Could not extract file path from upload", None, empty_df
        
        # Validate file exists and is readable
        if not isinstance(audio_file, str):
            return f"Invalid audio file format: {type(audio_file)}. Please upload a valid audio file.", None, empty_df
        
        # Check for invalid paths (like root directory) - this is the Modal bug
        if audio_file == "/" or audio_file == "" or audio_file == "." or audio_file == "/favicon.ico":
            return "Please upload an audio file", None, empty_df
        
        audio_path = Path(audio_file)
        
        # Debug logging for Modal
        if self.is_modal:
            import os
            try:
                stat_info = os.stat(audio_file)
                is_dir = os.path.isdir(audio_file)
                is_file = os.path.isfile(audio_file)
                # Log for debugging (won't show in production but helps identify issue)
                print(f"Modal debug - path: {audio_file}, is_dir: {is_dir}, is_file: {is_file}, exists: {audio_path.exists()}")
            except Exception as e:
                print(f"Modal debug - stat error: {e}")
        
        if not audio_path.exists():
            return f"Audio file not found: {audio_file}. The file may have been moved or deleted.", None, empty_df
        
        # Check if it's a directory (Modal/Gradio might return directory paths)
        # If it's a directory, try to find the actual audio file inside
        if audio_path.is_dir():
            # In Modal, Gradio might extract files to a directory
            # Look for audio files in the directory
            audio_extensions = {'.mp3', '.wav', '.flac', '.m4a'}
            audio_files_in_dir = [f for f in audio_path.iterdir() 
                                 if f.is_file() and f.suffix.lower() in audio_extensions]
            if len(audio_files_in_dir) == 1:
                # Use the single audio file found
                audio_file = str(audio_files_in_dir[0])
                audio_path = audio_files_in_dir[0]
                print(f"Modal: Found audio file in directory: {audio_file}")
            elif len(audio_files_in_dir) > 1:
                return f"Error: Directory contains multiple audio files. Please upload a single file. Found: {[f.name for f in audio_files_in_dir]}", None, empty_df
            else:
                return f"Error: Expected a file but got a directory with no audio files: {audio_file}. Please upload a valid audio file.", None, empty_df
        
        # Check file size (should be > 0)
        try:
            if not audio_path.is_file():
                return f"Error: Path is not a regular file: {audio_file}. Path exists: {audio_path.exists()}, is_dir: {audio_path.is_dir()}, is_file: {audio_path.is_file()}", None, empty_df
            if audio_path.stat().st_size == 0:
                return "Error: The uploaded file is empty. Please upload a valid audio file.", None, empty_df
        except OSError as e:
            return f"Error accessing file: {str(e)}", None, empty_df
        
        try:
            # Double-check it's a file before processing (Modal-specific issue)
            if not audio_path.is_file():
                return f"Error: Path is not a regular file: {audio_file}. Got type: {type(audio_path)}", None, empty_df
            
            # Preprocess audio file
            print(f"[{time.strftime('%H:%M:%S')}] Processing audio file: {audio_file}", flush=True)
            import sys
            sys.stdout.flush()
            preprocess_start = time.time()
            input_data = preprocess_audio(str(audio_file))
            preprocess_time = time.time() - preprocess_start
            print(f"[{time.strftime('%H:%M:%S')}] Audio preprocessing completed in {preprocess_time:.2f}s", flush=True)
            print(f"[{time.strftime('%H:%M:%S')}] Preprocessed tensor shape: {input_data.shape}, dtype: {input_data.dtype}", flush=True)
            sys.stdout.flush()
            
            if self.onnx:
                # ONNX inference
                start_time = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] Starting ONNX inference...")
                
                # input_data shape: [1, audio_length] -> squeeze to [audio_length] -> reshape to [1, audio_length]
                input_tensor = input_data.squeeze(0).numpy().reshape(1, -1).astype(np.float32)
                print(f"[{time.strftime('%H:%M:%S')}] Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
                
                # Verify input shape matches ONNX model expectations
                if self.onnx_session is None:
                    error_msg = "Error: ONNX session not initialized"
                    print(f"[{time.strftime('%H:%M:%S')}] {error_msg}", flush=True)
                    return error_msg, None, empty_df
                
                import sys
                sys.stdout.flush()
                
                # Get expected input shape from model
                try:
                    input_name = self.onnx_session.get_inputs()[0].name
                    expected_shape = self.onnx_session.get_inputs()[0].shape
                    print(f"[{time.strftime('%H:%M:%S')}] Model expects input '{input_name}' with shape: {expected_shape}", flush=True)
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] Warning: Could not get model input info: {e}", flush=True)
                
                print(f"[{time.strftime('%H:%M:%S')}] Running ONNX session.run()...", flush=True)
                sys.stdout.flush()
                
                try:
                    inference_start = time.time()
                    
                    # Ensure input is contiguous and correct shape
                    if not input_tensor.flags['C_CONTIGUOUS']:
                        input_tensor = np.ascontiguousarray(input_tensor)
                    
                    # Validate input shape matches expected
                    print(f"[{time.strftime('%H:%M:%S')}] Expected shape: {expected_shape}, Got: {input_tensor.shape}", flush=True)
                    
                    # Handle dynamic batch dimension
                    if len(expected_shape) == 2 and (expected_shape[0] == -1 or expected_shape[0] == 'batch_size'):
                        # Dynamic batch size - ensure we have batch dimension
                        if len(input_tensor.shape) == 1:
                            input_tensor = input_tensor.reshape(1, -1)
                        print(f"[{time.strftime('%H:%M:%S')}] Final input shape: {input_tensor.shape}", flush=True)
                    
                    # Run inference with explicit input name
                    print(f"[{time.strftime('%H:%M:%S')}] Calling session.run()...", flush=True)
                    print(f"[{time.strftime('%H:%M:%S')}] Active providers: {self.onnx_session.get_providers()}", flush=True)
                    sys.stdout.flush()
                    
                    # This is the critical call - if it hangs, we'll see it in logs
                    outputs = self.onnx_session.run(
                        ['binary_logit', 'tag_logits'],
                        {input_name: input_tensor}
                    )
                    inference_time = time.time() - inference_start
                    print(f"[{time.strftime('%H:%M:%S')}] ONNX inference completed in {inference_time:.2f}s", flush=True)
                    sys.stdout.flush()
                except Exception as e:
                    error_msg = f"ONNX inference failed: {str(e)}"
                    print(f"[{time.strftime('%H:%M:%S')}] {error_msg}")
                    print(f"[{time.strftime('%H:%M:%S')}] Traceback: {traceback.format_exc()}")
                    return error_msg, None, empty_df
                
                binary_logit, _ = outputs
                print(f"[{time.strftime('%H:%M:%S')}] Raw binary_logit: {binary_logit}, shape: {binary_logit.shape}")
                
                # Convert numpy scalar to Python float for consistency
                ai_prob = float(1 / (1 + np.exp(-binary_logit[0, 0])))
                total_time = time.time() - start_time
                print(f"[{time.strftime('%H:%M:%S')}] Total ONNX processing time: {total_time:.2f}s, AI probability: {ai_prob:.3f}")
            else:
                # PyTorch inference
                print("Starting PyTorch inference...")
                if self.model is None:
                    return "Error: PyTorch model not initialized", None, empty_df
                
                if device is None:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                input_tensor = input_data.to(device)
                ai_prob = predict_ai_only(self.model, input_tensor)
                # Ensure it's a Python float
                if isinstance(ai_prob, torch.Tensor):
                    ai_prob = ai_prob.item()
                ai_prob = float(ai_prob)
                print("Inference complete")
            
            is_ai = ai_prob > self.threshold
            result = f"**AI-Generated: {'Yes' if is_ai else 'No'}**\n"
            result += f"Confidence: {ai_prob:.3f} ({ai_prob*100:.1f}%)\n"
            result += f"Threshold: {self.threshold}"
                
            # Get music info
            try:
                music_info = analyze_music_simple(audio_file)
                result += f"\n\n**Music Analysis:**\n"
                result += f"Genre: {music_info['genre']}\n"
                result += f"Mood: {music_info['mood']}\n"
                result += f"Tempo: {music_info['tempo']} BPM\n"
                result += f"Energy: {music_info['energy']}"
                
                # Create dataframe for display
                df = pd.DataFrame([{
                    'Filename': Path(audio_file).name,
                    'AI-Generated': 'Yes' if is_ai else 'No',
                    'Confidence': f"{ai_prob:.3f}",
                    'Genre': music_info['genre'],
                    'Mood': music_info['mood'],
                    'Tempo (BPM)': music_info['tempo'],
                    'Energy': music_info['energy']
                }])
            except Exception as e:
                result += f"\n\n**Music Analysis:**\nError analyzing music: {str(e)}"
                # Create dataframe with error info
                df = pd.DataFrame([{
                    'Filename': Path(audio_file).name,
                    'AI-Generated': 'Yes' if is_ai else 'No',
                    'Confidence': f"{ai_prob:.3f}",
                    'Genre': 'Error',
                    'Mood': 'Error',
                    'Tempo (BPM)': 'Error',
                    'Energy': 'Error'
                }])
                
            return result, ai_prob, df
    
        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}\n\n{traceback.format_exc()}"
            # Return empty dataframe on error
            empty_df = pd.DataFrame(columns=['Filename', 'AI-Generated', 'Confidence', 'Genre', 'Mood', 'Tempo (BPM)', 'Energy'])
            return error_msg, None, empty_df
        
    
    def __predict_folder_batch(self, zip_file: str):
        """Predict on a zip file containing audio files"""
        if not zip_file:
            return (
                "Please upload a ZIP file",
                None,
                None,
                None,
                None
            )
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Handle Gradio File component - it returns a file path string or dict
                if isinstance(zip_file, dict):
                    file_path = zip_file.get('name', zip_file.get('path', None))
                else:
                    file_path = str(zip_file) if zip_file else None
                
                if not file_path:
                    return (
                        "Please upload a ZIP file",
                        None, None, None, None
                    )
                
                # Only accept zip files
                zip_path = Path(file_path)
                if not zip_path.exists():
                    return (
                        f"ZIP file not found: {file_path}",
                        None, None, None, None
                    )
                
                # Check if it's a directory (Modal/Gradio might return directory paths)
                if zip_path.is_dir():
                    return (
                        f"Error: Expected a ZIP file but got a directory: {file_path}. Please upload a ZIP file.",
                        None, None, None, None
                    )
                
                if not zip_path.is_file():
                    return (
                        f"Error: Path is not a regular file: {file_path}",
                        None, None, None, None
                    )
                
                if not zip_path.suffix.lower() == '.zip':
                    return (
                        f"Invalid file type. Please upload a ZIP file (got: {zip_path.suffix})",
                        None, None, None, None
                    )
                
                # Extract zip file
                extract_dir = temp_path / "extracted"
                extract_dir.mkdir()
                
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                except zipfile.BadZipFile:
                    return (
                        f"Invalid ZIP file: {file_path}. Please ensure the file is a valid ZIP archive.",
                        None, None, None, None
                    )
                
                folder_to_process = extract_dir
                
                audio_files = get_audio_files(str(folder_to_process))
                
                if len(audio_files) == 0:
                    return (
                        "No audio files found in the uploaded ZIP file", 
                        None, 
                        None, 
                        None,
                        None
                    )
                
                # Process folder using already-loaded model
                output_dir = temp_path / "results"
                output_dir.mkdir()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                csv_path = output_dir / f"predictions_{timestamp}.csv"
                
                # Import CSV writing functions
                from predict import write_music_header, append_music_row, write_final_accuracy_row
                
                write_music_header(str(csv_path))
                
                # Process each audio file
                for file_path in audio_files:
                    # Additional safety check: skip macOS metadata files
                    if file_path.name.startswith('._') or '__MACOSX' in str(file_path):
                        continue
                    
                    # Skip if not a regular file
                    if not file_path.is_file():
                        continue
                    
                    try:
                        input_data = preprocess_audio(str(file_path))
                        
                        if self.onnx:
                            # ONNX inference
                            input_tensor = input_data.squeeze(0).numpy().reshape(1, -1).astype(np.float32)
                            outputs = self.onnx_session.run(
                                ['binary_logit', 'tag_logits'],
                                {'audio': input_tensor}
                            )
                            binary_logit, _ = outputs
                            ai_prob = float(1 / (1 + np.exp(-binary_logit[0, 0])))
                        else:
                            # PyTorch inference
                            if device is None:
                                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            input_tensor = input_data.to(device)
                            ai_prob = predict_ai_only(self.model, input_tensor)
                        
                        ai_label = "Yes" if ai_prob > self.threshold else "No"
                        
                        music_info = analyze_music_simple(str(file_path))
                        
                        # If AI-generated, prefix the genre
                        if ai_label == "Yes":
                            music_info['genre'] = f"AI-{music_info['genre']}"
                        
                        # Write result with music tags
                        append_music_row(str(csv_path), file_path.name, ai_label, ai_prob, music_info)
                        
                    except Exception as e:
                        print(f"Error processing {file_path.name}: {e}")
                        # Write error row
                        error_info = {
                            'genre': 'Processing Error',
                            'mood': 'Unknown',
                            'instruments': 'Unknown', 
                            'tempo': 'Unknown',
                            'energy': 'Unknown',
                            'danceability': 'Unknown'
                        }
                        append_music_row(str(csv_path), file_path.name, "Unknown", 0.0, error_info)
                        continue
                
                # Write final accuracy row and create Excel file
                xlsx_path = write_final_accuracy_row(str(csv_path))
                
                # Read results
                df = pd.read_csv(csv_path)
                
                # Format for display - select key columns
                display_columns = ['filename', 'is_ai_generated', 'ai_confidence', 'genre', 'mood', 'tempo_bpm', 'energy']
                available_columns = [col for col in display_columns if col in df.columns]
                display_df = df[available_columns].copy()
                
                # Format confidence for display
                if 'ai_confidence' in display_df.columns:
                    display_df['ai_confidence'] = display_df['ai_confidence'].apply(
                        lambda x: f"{float(x):.3f}" if pd.notna(x) else "N/A"
                    )
                
                # Create summary
                summary_text = format_summary_stats(df)
                
                # Create visualizations
                fig = create_summary_visualizations(df)
                
                # Copy result file to a permanent location before temp directory is cleaned up
                # Use a persistent temp directory or copy to a known location
                persistent_dir = Path(tempfile.gettempdir()) / "gradio_audio_results"
                try:
                    persistent_dir.mkdir(exist_ok=True)
                except OSError as e:
                    # If directory creation fails (e.g., path exists as file), use a different approach
                    if self.is_modal:
                        # In Modal, use /tmp which is writable
                        persistent_dir = Path("/tmp") / "gradio_audio_results"
                        persistent_dir.mkdir(exist_ok=True)
                    else:
                        raise
                
                # Copy Excel file if it exists, otherwise copy CSV
                xlsx_path_obj = Path(xlsx_path)
                if xlsx_path_obj.exists() and xlsx_path_obj.is_file():
                    persistent_file = persistent_dir / xlsx_path_obj.name
                    shutil.copy2(xlsx_path_obj, persistent_file)
                    result_file = str(persistent_file)
                else:
                    # Fallback to CSV
                    csv_path_obj = Path(csv_path)
                    if csv_path_obj.exists() and csv_path_obj.is_file():
                        persistent_file = persistent_dir / csv_path_obj.name
                        shutil.copy2(csv_path_obj, persistent_file)
                        result_file = str(persistent_file)
                    else:
                        # Last resort: return the original path
                        result_file = str(csv_path)
                
                return (
                    summary_text,
                    display_df,
                    fig,
                    result_file,
                    f"Processed {len(df)} files successfully"
                )
        except Exception as e:
            error_msg = f"Error processing folder: {str(e)}\n\n{traceback.format_exc()}"
            return (
                error_msg,
                None,
                None,
                None,
                f"Error: {str(e)}"
            )
            
    def run_gradio(self):
        # Ensure Gradio is imported (in case import failed at module level)
        # Use a local variable to avoid shadowing module-level gr
        if not GRADIO_AVAILABLE or gr is None:
            try:
                import gradio
                gradio_module = gradio
            except ImportError as e:
                raise ImportError(f"Gradio is not available. Install with: pip install gradio. Error: {e}")
        else:
            gradio_module = gr
            
        # Monkey-patch Gradio's hash_file to prevent IsADirectoryError on Modal
        # This fixes the issue where Gradio tries to hash the root directory '/'
        try:
            from gradio import processing_utils
            if not hasattr(processing_utils, '_original_hash_file'):
                processing_utils._original_hash_file = processing_utils.hash_file
                
                def patched_hash_file(path, *args, **kwargs):
                    if str(path) == "/" or str(path) == "/favicon.ico":
                        return "root_bypass_hash"
                    try:
                        return processing_utils._original_hash_file(path, *args, **kwargs)
                    except (IsADirectoryError, PermissionError):
                        return "directory_bypass_hash"
                
                processing_utils.hash_file = patched_hash_file
                print("Applied fix for Gradio IsADirectoryError")
        except Exception as e:
            print(f"Could not patch Gradio: {e}")
        
        # Configure Gradio cache directory for Modal
        if self.is_modal:
            import os
            gradio_cache = "/tmp/gradio"
            os.makedirs(gradio_cache, exist_ok=True)
            # Set environment variable for Gradio
            os.environ["GRADIO_TEMP_DIR"] = gradio_cache
            # Try to set Gradio's cache directory directly
            try:
                if hasattr(gradio_module, 'utils'):
                    gradio_module.utils.GRADIO_CACHE = gradio_cache
                # Also try setting it on the File component class
                if hasattr(gradio_module.components, 'File'):
                    gradio_module.components.File.GRADIO_CACHE = gradio_cache
            except Exception:
                pass  # If setting fails, continue anyway
        
        # Create Gradio Blocks - ensure it's properly initialized
        demo = gradio_module.Blocks(
            title="AI Audio Detection", 
            theme=gradio_module.themes.Soft()
        )
        
        # Configure queue for long-running requests
        # Note: Modal's ASGI wrapper handles concurrency, so we use a higher limit
        try:
            demo.queue(
                default_concurrency_limit=10,  # Higher limit - Modal handles actual concurrency
                max_size=50,  # Allow more queued requests
                api_open=False  # Don't expose queue API
            )
            print("Gradio queue configured successfully")
        except Exception as e:
            print(f"Warning: Could not configure Gradio queue: {e}")
            # Continue anyway - queue might not be available in all Gradio versions
        
        with demo:
            gradio_module.Markdown("""
                # AI-Generated Audio Detection
                # Upload audio files to detect if they're AI-generated. Supports single files or batch folder processing with detailed analysis.
            """)
        
            with gradio_module.Tabs():
                with gradio_module.Tab("Single File"):
                    with gradio_module.Row():
                        with gradio_module.Column():
                            gradio_module.Markdown("### Upload an audio file (MP3, WAV, FLAC, M4A)")
                            audio_input = gradio_module.File(
                                label="Upload Audio File",
                                file_types=[".mp3", ".wav", ".flac", ".m4a"],  # Explicit extensions instead of ["audio"]
                                file_count="single",
                                value=None,  # Explicitly set to None to avoid Modal's root directory bug
                                show_label=True,
                                interactive=True
                            )
                            predict_btn = gradio_module.Button("Detect", variant="primary", size="lg")
                        
                        with gradio_module.Column():
                            gradio_module.Markdown("### Prediction Result")
                            output_text = gradio_module.Markdown("")
                            confidence_bar = gradio_module.Number(
                                label="AI Probability",
                                visible=False
                            )
                            results_df = gradio_module.Dataframe(
                                label="Results Table",
                                headers=["Filename", "AI-Generated", "Confidence", "Genre", "Mood", "Tempo (BPM)", "Energy"],
                                interactive=False,
                                height=200
                            )
                    
                    def safe_predict(audio_file):
                        """Wrapper to catch Gradio preprocessing errors and Modal's root directory bug"""
                        import sys
                        # Log to both stdout and stderr for maximum visibility in Modal
                        log_msg = f"[{time.strftime('%H:%M:%S')}] ===== safe_predict CALLED ====="
                        print(log_msg, flush=True)
                        print(log_msg, file=sys.stderr, flush=True)
                        sys.stdout.flush()
                        sys.stderr.flush()
                        
                        log_msg = f"[{time.strftime('%H:%M:%S')}] audio_file type: {type(audio_file)}, value: {audio_file}"
                        print(log_msg, flush=True)
                        print(log_msg, file=sys.stderr, flush=True)
                        sys.stdout.flush()
                        sys.stderr.flush()
                        
                        # Early filter for Modal's root directory bug
                        if audio_file is None:
                            print(f"[{time.strftime('%H:%M:%S')}] audio_file is None, returning early", flush=True)
                            return self.__predict_single_audio(None)
                        # Handle list inputs (Modal/Gradio might return lists)
                        if isinstance(audio_file, list):
                            if len(audio_file) == 0:
                                print(f"[{time.strftime('%H:%M:%S')}] audio_file is empty list, returning early", flush=True)
                                return self.__predict_single_audio(None)
                            audio_file = audio_file[0]
                            print(f"[{time.strftime('%H:%M:%S')}] Extracted from list: {audio_file}", flush=True)
                        # Filter out root directory and favicon (Modal bug)
                        # Also check if it's a directory
                        if isinstance(audio_file, str):
                            if audio_file == "/" or audio_file == "/favicon.ico" or audio_file == "" or audio_file == ".":
                                print(f"[{time.strftime('%H:%M:%S')}] Invalid path detected: {audio_file}, returning early", flush=True)
                                return self.__predict_single_audio(None)
                            # Check if it's a directory path
                            try:
                                import os
                                if os.path.isdir(audio_file):
                                    print(f"[{time.strftime('%H:%M:%S')}] Path is directory: {audio_file}, returning early", flush=True)
                                    return self.__predict_single_audio(None)
                            except Exception as e:
                                print(f"[{time.strftime('%H:%M:%S')}] Error checking if directory: {e}", flush=True)
                                pass  # If check fails, continue
                        
                        log_msg = f"[{time.strftime('%H:%M:%S')}] Calling __predict_single_audio with: {audio_file}"
                        print(log_msg, flush=True)
                        print(log_msg, file=sys.stderr, flush=True)
                        sys.stdout.flush()
                        sys.stderr.flush()
                        
                        try:
                            result = self.__predict_single_audio(audio_file)
                            log_msg = f"[{time.strftime('%H:%M:%S')}] ===== safe_predict completed successfully ====="
                            print(log_msg, flush=True)
                            print(log_msg, file=sys.stderr, flush=True)
                            sys.stdout.flush()
                            sys.stderr.flush()
                            return result
                        except (IsADirectoryError, OSError) as e:
                            # Handle directory errors gracefully
                            print(f"[{time.strftime('%H:%M:%S')}] Directory error caught: {e}", flush=True)
                            if "Is a directory" in str(e) or "IsADirectoryError" in str(type(e).__name__):
                                return self.__predict_single_audio(None)
                            raise
                        except Exception as e:
                            error_msg = f"Error processing audio file: {str(e)}\n\n"
                            error_msg += "This may be due to:\n"
                            error_msg += "- Unsupported audio format\n"
                            error_msg += "- Corrupted audio file\n"
                            error_msg += "- Missing audio codecs\n\n"
                            error_msg += f"Technical details: {traceback.format_exc()}"
                            print(f"[{time.strftime('%H:%M:%S')}] Exception in safe_predict: {error_msg}", flush=True)
                            sys.stdout.flush()
                            empty_df = pd.DataFrame(columns=['Filename', 'AI-Generated', 'Confidence', 'Genre', 'Mood', 'Tempo (BPM)', 'Energy'])
                            return error_msg, None, empty_df
                    
                    predict_btn.click(
                        fn=safe_predict,
                        inputs=audio_input,
                        outputs=[output_text, confidence_bar, results_df]
                    )
                
                with gradio_module.Tab("Batch Processing"):
                    gradio_module.Markdown("### Upload a zip file containing multiple audio files")
                    gradio_module.Markdown("**Note:** Only ZIP files are supported. Please zip your audio files before uploading.")
                    
                    with gradio_module.Row():
                        folder_input = gradio_module.File(
                            label="Upload ZIP File",
                            file_count="single",
                            file_types=[".zip"],
                            value=None  # Explicitly set to None
                        )
                        batch_predict_btn = gradio_module.Button("Process ZIP File", variant="primary", size="lg")
                    
                    status_text = gradio_module.Markdown("")
                    
                    with gradio_module.Row():
                        with gradio_module.Column(scale=1):
                            gradio_module.Markdown("### Summary Statistics")
                            summary_markdown = gradio_module.Markdown("Upload a ZIP file and click 'Process ZIP File' to see results.")
                        
                        with gradio_module.Column(scale=1):
                            gradio_module.Markdown("### Visualizations")
                            results_plot = gradio_module.Plot()
                    
                    results_table = gradio_module.Dataframe(
                        label="Detailed Results",
                        headers=["Filename", "AI-Generated", "Confidence", "Genre", "Mood", "Tempo", "Energy"],
                        interactive=False,
                        height=400
                    )
                    
                    with gradio_module.Row():
                        result_file_download = gradio_module.File(
                            label="Download Full Results (CSV/Excel)",
                            value=None,  # Explicitly set to None
                            interactive=False  # Output only
                        )
                    
                    batch_predict_btn.click(
                        fn=self.__predict_folder_batch,
                        inputs=folder_input,
                        outputs=[
                            summary_markdown,
                            results_table,
                            results_plot,
                            result_file_download,
                            status_text
                        ])
            
            gradio_module.Markdown("""
            ---
            ### Usage Tips
            
            - **Single File**: Upload one audio file (.mp3, .wav, .flac, .m4a) for instant prediction
            - **Batch Processing**: Upload a zip file containing multiple audio files
            - Results include AI detection confidence and music analysis (genre, mood, tempo, energy)
            - Batch processing shows interactive tables, charts, and summary statistics
            - Download full results as CSV/Excel for further analysis
            """)
        
        # Ensure the Blocks object is fully built and ready
        # The context manager should have completed, but let's make sure
        if not hasattr(demo, '__call__'):
            raise RuntimeError("Gradio Blocks object is not properly initialized as an ASGI app")
        
        # Set max_file_size attribute manually since it's not a constructor arg but required by upload route
        # Use a large value (e.g., 1GB) or check if None works (often means unlimited)
        # The error "AttributeError: 'Blocks' object has no attribute 'max_file_size'" confirms this is needed
        if not hasattr(demo, 'max_file_size'):
            demo.max_file_size = 1024 * 1024 * 1024  # 1 GB limit
    
        # Set root_path to empty string for Modal to prevent '/' path issues
        # This prevents Gradio from trying to process '/' as a file path on page load
        if hasattr(demo, 'root_path'):
            demo.root_path = ""
        elif hasattr(demo, 'config'):
            # Try setting it via config if available
            if hasattr(demo.config, 'root_path'):
                demo.config.root_path = ""
    
        return demo
    
    def run_local(self):
        """Run Gradio app locally."""
        demo = self.run_gradio()
        print("Launching Gradio interface locally...")
        demo.launch(
            server_name="127.0.0.1", 
            server_port=7860, 
            share=False,
            show_api=False  # Disable API docs to avoid schema generation issues
        )

def format_summary_stats(df):
    """Format summary statistics as Markdown."""
    total_files = len(df)
    if total_files == 0:
        return "### Summary Statistics\n\nNo files processed."
    
    ai_count = len(df[df['is_ai_generated'] == 'Yes'])
    real_count = len(df[df['is_ai_generated'] == 'No'])
    df['ai_confidence'] = pd.to_numeric(df['ai_confidence'], errors='coerce')
    avg_confidence = df['ai_confidence'].mean()
    
    summary = f"""
### Summary Statistics

**Overall Results:**
- **Total Files Processed:** {total_files}
- **AI-Generated:** {ai_count} ({ai_count/total_files*100:.1f}%)
- **Real Music:** {real_count} ({real_count/total_files*100:.1f}%)
- **Average Confidence:** {avg_confidence:.3f}
"""
    return summary

def create_summary_visualizations(df):
    """Create visualization plots for batch results."""
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Pie chart: AI vs Real
    ai_count = len(df[df['is_ai_generated'] == 'Yes'])
    real_count = len(df[df['is_ai_generated'] == 'No'])
    axes[0].pie([real_count, ai_count], 
                labels=['Real', 'AI-Generated'],
                autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'])
    axes[0].set_title('AI vs Real Distribution')
    
    # Confidence histogram
    df['ai_confidence'] = pd.to_numeric(df['ai_confidence'], errors='coerce')
    axes[1].hist(df['ai_confidence'], bins=20, color='#2196F3', edgecolor='black', alpha=0.7)
    axes[1].axvline(DEFAULT_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({DEFAULT_THRESHOLD})')
    axes[1].set_xlabel('AI Confidence Score')
    axes[1].set_ylabel('Number of Files')
    axes[1].set_title('Confidence Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

if MODAL_AVAILABLE:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libsndfile1")
        .pip_install(
            "gradio>=4.44.1,<5.0.0",  # Latest Gradio 4.x (4.44.1) for Python 3.11 - compatible with huggingface_hub<0.23.0
            "huggingface_hub>=0.20.0,<0.23.0",  # Compatible with Gradio 4.44.1 (HfFolder was removed in 0.23.0+)
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "onnxruntime>=1.23.2",  # Latest ONNX Runtime for Python 3.11 (Nov 2024)
            "numpy>=1.24.0",
            "librosa>=0.11.0",
            "soundfile>=0.12.0",
            "pandas>=2.0.0",
            "openpyxl>=3.1.0",
            "matplotlib>=3.9.0",
            "torchlibrosa>=0.1.0",
            "scipy>=1.13.1",
            "scikit-learn>=1.6.1",
            "tqdm>=4.67.1",
        )
        .add_local_dir(".", remote_path="/root", copy=True, ignore=[
            "*.pth", "*.pth.gz", "*.onnx",  # Exclude model files
            "model/saved_models/", "data/",  # Exclude model and data directories
            ".venv/", "__pycache__/", ".git/",  # Exclude build artifacts
        ])
        .run_commands("cd /root && pip install -e .")
    )
    
    app = modal.App("ai-audio-detection")
    model_volume = modal.Volume.from_name("ai-audio-models", create_if_missing=True)
    
    @app.function(
        image=image,
        volumes={"/models": model_volume},
        timeout=600,  # Increased timeout to 10 minutes
        scaledown_window=300,  # Keep container alive for 5 minutes (renamed from container_idle_timeout)
        # Removed gpu="any" since we're using CPU for ONNX inference
    )
    @modal.asgi_app()  # Outermost decorator - ASGI apps handle concurrency internally
    def gradio_app_modal():
        """Modal deployment function - uses ONNX model from volume."""
        import sys
        import os
        sys.path.insert(0, "/root")
        
        # Create Gradio cache directory if it doesn't exist (Modal-specific fix)
        gradio_cache_dir = "/tmp/gradio"
        os.makedirs(gradio_cache_dir, exist_ok=True)
        
        # Set environment variable for Gradio cache
        os.environ["GRADIO_TEMP_DIR"] = gradio_cache_dir
        
        ONNX_MODEL_PATH = "/models/model.onnx"
        THRESHOLD = DEFAULT_THRESHOLD
        
        # Check if model exists, if not provide helpful error
        if not os.path.exists(ONNX_MODEL_PATH):
            raise FileNotFoundError(
                f"ONNX model not found at {ONNX_MODEL_PATH}. "
                "Please upload your model to the Modal volume first using:\n"
                "modal volume put ai-audio-models /path/to/your/model.onnx model.onnx"
            )
        
        interface = GradioAudioInterface(
            model_path=ONNX_MODEL_PATH,
            threshold=THRESHOLD,
            use_onnx=True,
            is_modal=True
        )
        demo = interface.run_gradio()
        
        # In Gradio 4.x, Blocks implements the ASGI interface directly
        # Set root_path to empty string to prevent '/' path issues in Modal
        # This is similar to setting root_path="" in demo.launch() but for ASGI deployment
        try:
            if hasattr(demo, 'root_path'):
                demo.root_path = ""
            # Also try setting via config if available
            if hasattr(demo, 'config') and hasattr(demo.config, 'root_path'):
                demo.config.root_path = ""
        except Exception:
            pass  # If setting fails, continue anyway - not critical
        
        # Return the ASGI-compatible app
        # Gradio Blocks are ASGI-compatible, but demo.app (FastAPI) is more explicit
        if hasattr(demo, "app"):
            return demo.app  # FastAPI instance - preferred for Modal
        elif callable(demo):
            return demo  # Gradio Blocks are also ASGI-compatible
        else:
            raise RuntimeError(f"Expected an ASGI app, but got {type(demo)}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio app for AI audio detection")
    parser.add_argument(
        "--model",
        help="Path to model file (.pth or .onnx)",
        required=True,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Threshold for AI detection (default: 0.35)",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Use ONNX model instead of PyTorch",
    )
    
    args = parser.parse_args()
    gradio_interface = GradioAudioInterface(
        model_path=args.model,
        threshold=args.threshold,
        use_onnx=args.onnx
    )
    gradio_interface.run_local()
