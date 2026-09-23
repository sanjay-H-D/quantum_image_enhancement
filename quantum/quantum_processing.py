import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def quantum_inspired_processing(image):
    # Normalize pixel values
    normalized = image.astype(np.float32) / 255.0

    # Quantum-inspired transformation
    processed = np.sqrt(normalized)

    # Convert back to 0-255
    processed = (processed * 255).astype(np.uint8)

    return processed


image_names = ["brain1", "brain2", "brain3"]

original_images = []
processed_images = []


for name in image_names:

    # Automatically find the image
    possible_extensions = [".jpeg", ".jpg", ".png"]

    image_path = None

    for extension in possible_extensions:
        path = f"dataset/images/{name}{extension}"

        if os.path.exists(path):
            image_path = path
            break

    if image_path is None:
        print(f"ERROR: {name} image not found")
        continue

    print(f"\nReading: {image_path}")

    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"ERROR: Could not read {name}")
        continue

    # Process
    processed = quantum_inspired_processing(image)

    # Save
    output_path = f"dataset/processed/{name}_quantum.png"

    success = cv2.imwrite(output_path, processed)

    if success:
        print(f"{name} processed successfully")
        print(f"Saved: {output_path}")
    else:
        print(f"ERROR: Could not save {name}")

    original_images.append(image)
    processed_images.append(processed)


# Display
plt.figure(figsize=(12, 8))

for i in range(len(original_images)):

    plt.subplot(2, 3, i + 1)
    plt.imshow(original_images[i], cmap="gray")
    plt.title(f"{image_names[i]} - Original")
    plt.axis("off")

    plt.subplot(2, 3, i + 4)
    plt.imshow(processed_images[i], cmap="gray")
    plt.title(f"{image_names[i]} - Quantum-Inspired")
    plt.axis("off")

plt.tight_layout()
plt.show()