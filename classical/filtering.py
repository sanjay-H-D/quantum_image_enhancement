import cv2
import matplotlib.pyplot as plt

# Read the three MRI images
img1 = cv2.imread("dataset/images/brain1.jpeg")
img2 = cv2.imread("dataset/images/brain2.jpeg")
img3 = cv2.imread("dataset/images/brain3.jpeg")

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Apply Gaussian filtering
filtered1 = cv2.GaussianBlur(gray1, (5, 5), 0)
filtered2 = cv2.GaussianBlur(gray2, (5, 5), 0)
filtered3 = cv2.GaussianBlur(gray3, (5, 5), 0)

# Display original and filtered images
plt.figure(figsize=(12, 8))

# Original
plt.subplot(2, 3, 1)
plt.imshow(gray1, cmap="gray")
plt.title("Brain 1 - Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray2, cmap="gray")
plt.title("Brain 2 - Original")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(gray3, cmap="gray")
plt.title("Brain 3 - Original")
plt.axis("off")

# Filtered
plt.subplot(2, 3, 4)
plt.imshow(filtered1, cmap="gray")
plt.title("Brain 1 - Filtered")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(filtered2, cmap="gray")
plt.title("Brain 2 - Filtered")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(filtered3, cmap="gray")
plt.title("Brain 3 - Filtered")
plt.axis("off")

plt.tight_layout()
plt.show()