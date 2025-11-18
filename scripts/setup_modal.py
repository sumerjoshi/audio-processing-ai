#!/usr/bin/env python3
"""
Script to convert PyTorch model to ONNX and upload to Modal volume.
Run this before deploying to Modal.
"""

import argparse
import sys
from pathlib import Path

try:
    import modal
except ImportError:
    print("Error: modal is not installed. Install with: pip install modal")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("Error: torch is not installed. Install with: pip install torch")
    sys.exit(1)

try:
    import onnx
except ImportError:
    print("Error: onnx is not installed. Install with: pip install onnx")
    print("Note: onnx is required for torch.onnx.export()")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime is not installed. Install with: pip install onnxruntime")
    sys.exit(1)

# Import model architecture
sys.path.insert(0, str(Path(__file__).parent.parent))
from predict import DualHeadCnn14Simple, preprocess_audio
import torch

def convert_to_onnx(pth_path: str, onnx_path: str, sample_rate: int = 16000, duration: float = 10.0):
    """Convert PyTorch model to ONNX format."""
    print(f"Loading PyTorch model from: {pth_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadCnn14Simple(pretrained=False)
    model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=False))
    model.eval()
    
    # Create dummy input (batch_size=1, audio_length)
    audio_length = int(sample_rate * duration)
    dummy_input = torch.randn(1, audio_length).to(device)
    
    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["audio"],
        output_names=["binary_logit", "tag_logits"],
        dynamic_axes={
            "audio": {0: "batch_size"},
            "binary_logit": {0: "batch_size"},
            "tag_logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    
    print(f"ONNX model saved to: {onnx_path}")
    
    print("Verifying ONNX model...")
    session = ort.InferenceSession(onnx_path)
    print(f"ONNX model verified! Input shape: {session.get_inputs()[0].shape}")
    
    return onnx_path

def upload_to_modal(onnx_path: str, volume_name: str = "ai-audio-models", remote_path: str = "model.onnx"):
    """Upload ONNX model to Modal volume using Modal CLI."""
    import subprocess
    import os
    
    print(f"Uploading {onnx_path} to Modal volume '{volume_name}' as '{remote_path}'...")
    
    # Use Modal CLI to upload the file
    cmd = [
        "modal", "volume", "put",
        volume_name,
        onnx_path,
        remote_path
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"   Upload successful!")
        print(f"  Volume: {volume_name}")
        print(f"  Remote path: {remote_path}")
        print(f"  Use this path in your Modal app: /models/{remote_path}")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else ""
        
        # Check if it's an authentication error
        if "Token missing" in error_msg or "Could not authenticate" in error_msg:
            print(f"\nModal authentication required!")
            print(f"  Please authenticate with Modal first:")
            print(f"  1. Run: modal token new")
            print(f"  2. Then upload manually with:")
            print(f"    modal volume put {volume_name} {onnx_path} {remote_path}")
            print(f"\n  Or re-run this script after authenticating.")
        else:
            print(f"Error uploading to Modal: {error_msg}")
            print(f"\nYou can manually upload with:")
            print(f"  modal volume put {volume_name} {onnx_path} {remote_path}")
        
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX and upload to Modal")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PyTorch model file (.pth)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model.onnx",
        help="Output ONNX file path (default: model.onnx)"
    )
    parser.add_argument(
        "--volume",
        type=str,
        default="ai-audio-models",
        help="Modal volume name (default: ai-audio-models)"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading to Modal (just convert to ONNX)"
    )
    
    args = parser.parse_args()
    
    # Convert to ONNX
    onnx_path = convert_to_onnx(args.model, args.output)
    
    # Upload to Modal
    if not args.skip_upload:
        upload_success = upload_to_modal(onnx_path, args.volume)
        if upload_success:
            print("\nSetup complete! You can now deploy to Modal with:")
            print("   modal deploy gradio_app.py")
        else:
            print("\nONNX conversion complete!")
            print("   Upload to Modal failed. Please authenticate and upload manually:")
            print(f"      modal volume put {args.volume} {onnx_path} model.onnx")
            print("   Then deploy with:")
            print("      modal deploy gradio_app.py")
    else:
        print("\nONNX conversion complete!")
        print(f"   Upload manually with: modal volume put {args.volume} {onnx_path} model.onnx")

if __name__ == "__main__":
    main()

