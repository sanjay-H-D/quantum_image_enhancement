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

# Calculate histograms
hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([gray3], [0], None, [256], [0, 256])

# Display images
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(gray1, cmap="gray")
plt.title("Brain 1")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray2, cmap="gray")
plt.title("Brain 2")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(gray3, cmap="gray")
plt.title("Brain 3")
plt.axis("off")

# Display histograms
plt.subplot(2, 3, 4)
plt.plot(hist1)
plt.title("Brain 1 Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")

plt.subplot(2, 3, 5)
plt.plot(hist2)
plt.title("Brain 2 Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")

plt.subplot(2, 3, 6)
plt.plot(hist3)
plt.title("Brain 3 Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Number of Pixels")

plt.tight_layout()
plt.show()