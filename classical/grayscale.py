import cv2
import matplotlib.pyplot as plt

# Read the three MRI images
img1 = cv2.imread("dataset/images/brain1.jpeg")
img2 = cv2.imread("dataset/images/brain2.jpeg")
img3 = cv2.imread("dataset/images/brain3.jpeg")

# Check whether images were loaded
if img1 is None or img2 is None or img3 is None:
    print("Error: Could not load one or more images.")
    exit()

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

print("Brain 1 shape:", gray1.shape)
print("Brain 2 shape:", gray2.shape)
print("Brain 3 shape:", gray3.shape)