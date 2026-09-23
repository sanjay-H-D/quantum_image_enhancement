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

# Apply Histogram Equalization
equalized1 = cv2.equalizeHist(gray1)
equalized2 = cv2.equalizeHist(gray2)
equalized3 = cv2.equalizeHist(gray3)

# Display original and equalized images
plt.figure(figsize=(12, 8))

# Original images
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

# Equalized images
plt.subplot(2, 3, 4)
plt.imshow(equalized1, cmap="gray")
plt.title("Brain 1 - Equalized")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(equalized2, cmap="gray")
plt.title("Brain 2 - Equalized")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(equalized3, cmap="gray")
plt.title("Brain 3 - Equalized")
plt.axis("off")

plt.tight_layout()
plt.show()