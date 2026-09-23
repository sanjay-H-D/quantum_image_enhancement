import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_metrics(original, processed):
    mse = np.mean(
        (original.astype(float) - processed.astype(float)) ** 2
    )

    psnr = cv2.PSNR(original, processed)

    ssim_value = ssim(
        original,
        processed,
        data_range=255
    )

    return mse, psnr, ssim_value


for name in ["brain1", "brain2", "brain3"]:

    original = cv2.imread(
        f"dataset/images/{name}.jpeg",
        cv2.IMREAD_GRAYSCALE
    )

    processed = cv2.imread(
        f"dataset/processed/{name}_quantum.png",
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:
        print(f"{name}: Original image not found")
        continue

    if processed is None:
        print(f"{name}: Processed image not found")
        continue

    mse, psnr, ssim_value = calculate_metrics(
        original,
        processed
    )

    print(f"\n{name.upper()}")
    print("-------------------------")
    print(f"MSE  : {mse:.4f}")
    print(f"PSNR : {psnr:.4f} dB")
    print(f"SSIM : {ssim_value:.4f}")