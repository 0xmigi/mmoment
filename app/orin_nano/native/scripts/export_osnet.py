#!/usr/bin/env python3
"""
Export OSNet x0.25 to ONNX for ReID
Requires: pip install torchreid torch

Run this in a Python environment with PyTorch:
python3 export_osnet.py

Then convert to TensorRT using trtexec (in the l4t-tensorrt container):
/usr/src/tensorrt/bin/trtexec --onnx=osnet_x0_25.onnx --saveEngine=osnet_x0_25.engine \
    --fp16 --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw
"""

import torch
import os

def main():
    # OSNet x0.25 input: 256x128 (HxW), output: 512-dim embedding
    print("Downloading OSNet x0.25 pretrained model...")

    try:
        # Try torchreid first (official)
        import torchreid
        model = torchreid.models.build_model(
            name='osnet_x0_25',
            num_classes=1,  # Not used for feature extraction
            pretrained=True,
            use_gpu=torch.cuda.is_available()
        )
        model.eval()
    except ImportError:
        print("torchreid not available, downloading weights directly...")

        # Alternative: download weights and build model manually
        import urllib.request

        # OSNet x0.25 from torchreid model zoo (trained on ImageNet + Market1501)
        url = "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.6/osnet_x0_25_market1501_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip_jitter.pth"
        weights_path = "osnet_x0_25_market1501.pth"

        if not os.path.exists(weights_path):
            print(f"Downloading weights from {url}...")
            urllib.request.urlretrieve(url, weights_path)

        # Build OSNet manually - simplified version for feature extraction
        from osnet_model import OSNet, osnet_x0_25
        model = osnet_x0_25(num_classes=751, pretrained=False, loss='softmax')

        checkpoint = torch.load(weights_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model.eval()

    # Export to ONNX
    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, 256, 128)  # Batch x Channels x Height x Width

    # Use feature extraction mode (remove classifier)
    model.classifier = torch.nn.Identity()

    torch.onnx.export(
        model,
        dummy_input,
        "osnet_x0_25.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # Fixed batch size of 1
        opset_version=11,
        do_constant_folding=True
    )

    print("Exported osnet_x0_25.onnx")
    print("\nNow convert to TensorRT with:")
    print("/usr/src/tensorrt/bin/trtexec --onnx=osnet_x0_25.onnx --saveEngine=osnet_x0_25.engine --fp16")

if __name__ == "__main__":
    main()
