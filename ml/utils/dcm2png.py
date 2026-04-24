"""
dcm2png.py — Standalone DICOM to PNG converter.
Usage: python dcm2png.py <input.dcm> <output.png>
"""
import sys
import numpy as np
import cv2
import pydicom
from PIL import Image


def convert_dicom_to_png(dcm_path: str, png_path: str):
    """Convert a DICOM file to PNG with CLAHE enhancement."""
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)

    # Normalize to uint8
    if arr.max() > 0:
        arr = (arr / arr.max() * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    # CLAHE for better visualization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)

    # Save as PNG (grayscale → RGB)
    pil_img = Image.fromarray(arr, mode='L').convert('RGB')
    pil_img.save(png_path, format='PNG')
    print(f"OK: {png_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input.dcm> <output.png>", file=sys.stderr)
        sys.exit(1)

    convert_dicom_to_png(sys.argv[1], sys.argv[2])
