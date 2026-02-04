import cv2
import numpy as np
from scipy.ndimage import convolve, sobel
import matplotlib.pyplot as plt

# --- Weighted Mean Curvature Function ---
def weighted_mean_curvature(u):
    """
    Computes the Weighted Mean Curvature (Hw) of an input image or matrix u.
    :param u: 2D numpy array (grayscale image or matrix).
    :return: 2D numpy array representing the Weighted Mean Curvature (Hw).
    """
    k1 = np.array([[1, 1, 0], [2, -6, 0], [1, 1, 0]]) / 6
    k2 = np.array([[2, 4, 1], [4, -12, 0], [1, 0, 0]]) / 12

    dist = np.zeros((u.shape[0], u.shape[1], 8), dtype=np.float32)

    dist[:, :, 0] = convolve(u, k1, mode='constant', cval=0)
    dist[:, :, 1] = convolve(u, np.fliplr(k1), mode='constant', cval=0)
    dist[:, :, 2] = convolve(u, np.flipud(k1), mode='constant', cval=0)
    dist[:, :, 3] = convolve(u, np.fliplr(np.flipud(k1)), mode='constant', cval=0)
    dist[:, :, 4] = convolve(u, k2, mode='constant', cval=0)
    dist[:, :, 5] = convolve(u, np.fliplr(k2), mode='constant', cval=0)
    dist[:, :, 6] = convolve(u, np.flipud(k2), mode='constant', cval=0)
    dist[:, :, 7] = convolve(u, np.rot90(k2, 2), mode='constant', cval=0)

    abs_dist = np.abs(dist)
    min_indices = np.argmin(abs_dist, axis=2)
    Hw = np.choose(min_indices, dist.transpose(2, 0, 1))
    
    return Hw

# --- Structure Tensor Calculation ---
def structure_tensor(image, sigma=1.0):
    """
    Computes the Structure Tensor of an image.
    :param image: Grayscale image (n x m single-channel matrix).
    :param sigma: Gaussian smoothing factor.
    :return: Structure Tensor components (Jxx, Jyy, Jxy).
    """
    Ix = sobel(image, axis=1, mode='reflect')
    Iy = sobel(image, axis=0, mode='reflect')
    Ix2 = convolve(Ix**2, np.ones((3, 3)) / 9, mode='constant')
    Iy2 = convolve(Iy**2, np.ones((3, 3)) / 9, mode='constant')
    Ixy = convolve(Ix * Iy, np.ones((3, 3)) / 9, mode='constant')
    return Ix2, Iy2, Ixy

# --- Local Energy Function ---
def local_energy(Ix2, Iy2, Ixy):
    """
    Computes the Local Energy Function from Structure Tensor components.
    :return: Local energy image.
    """
    trace = Ix2 + Iy2
    determinant = (Ix2 * Iy2) - (Ixy**2)
    energy = trace - np.sqrt(np.maximum(trace**2 - 4 * determinant, 0))
    return energy

# --- XDoG Filter ---
def xdog(image, epsilon=0.01, k=200, gamma=0.98, phi=10):
    """
    Computes the eXtended Difference of Gaussians (XDoG).
    :param image: Grayscale image.
    """
    s1, s2 = 0.5, 0.5 * k
    gauss1 = convolve(image, np.ones((3, 3)) / 9, mode='constant')
    gauss2 = gamma * convolve(image, np.ones((5, 5)) / 25, mode='constant')
    dog_result = gauss1 - gauss2
    dog_result = dog_result / 255.0
    mask = dog_result >= epsilon
    dog_result[mask] = 1.0
    dog_result[~mask] = 1.0 + np.tanh(phi * (dog_result[~mask] - epsilon))
    return (dog_result * 255).astype(np.uint8)

# --- High-Frequency Fusion ---
def apply_structure_tensor_and_energy(A_H, B_H):

    """
    Fuses high-frequency components A_H and B_H using Structure Tensor, Local Energy, and XDoG for edge detection.
    """
    # Apply XDoG to the high-frequency components
    A_H_xdog = xdog(A_H)
    B_H_xdog = xdog(B_H)
    
    # Normalize the XDoG outputs to ensure they're in the correct range
    A_H_xdog = A_H_xdog.astype('float32') / 255.0
    B_H_xdog = B_H_xdog.astype('float32') / 255.0

    # Compute Structure Tensor and Local Energy on the XDoG-processed images
    Ix2_A, Iy2_A, Ixy_A = structure_tensor(A_H_xdog, sigma=1.0)
    Ix2_B, Iy2_B, Ixy_B = structure_tensor(B_H_xdog, sigma=1.0)

    energy_A = local_energy(Ix2_A, Iy2_A, Ixy_A)
    energy_B = local_energy(Ix2_B, Iy2_B, Ixy_B)

    DM_A = (energy_A > energy_B).astype(np.float32)  # Decision Map for A_H.
    DM_B = 1.0 - DM_A                                # Decision Map for B_H.
    
    F_H = DM_A * A_H_xdog + DM_B * B_H_xdog         # Fusion of high-frequency components.
    F_H = cv2.normalize(F_H, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return F_H

# --- Low-Frequency Fusion ---
def low_frequency_fusion(AL, BL):
    FL = (AL + BL) / 2.0
    return cv2.normalize(FL, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# --- Plot Function ---
def plot_images(images, titles, cmap='gray'):
    plt.figure(figsize=(15, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    # --- COA Plotting Function ---
def plot_coa(coyotes, costs):
    plt.figure(figsize=(10, 5))
    plt.plot(costs, marker='o')
    plt.title('COA Optimization Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()

# --- Main Processing Pipeline ---
IMRI = cv2.imread('C:/Program Files (x86)/USTH/DIP/MRI1.png', cv2.IMREAD_GRAYSCALE).astype('float32')
IPET = cv2.imread('C:/Program Files (x86)/USTH/DIP/PET.png', cv2.IMREAD_COLOR)

if IMRI is None or IPET is None:
    raise FileNotFoundError("One or both input images not found!")

A = cv2.convertScaleAbs(IMRI, alpha=1.2, beta=30).astype('float32')
PET_YUV = cv2.cvtColor(IPET, cv2.COLOR_BGR2YUV)
B = PET_YUV[:, :, 0].astype('float32')

# Low and High Frequency Decomposition using WMCF
AL = weighted_mean_curvature(A)
AH = A - AL
BL = weighted_mean_curvature(B)
BH = B - BL

# Plot Low-Frequency Components (WMCF)
plot_images([AL, BL], ["Low-Frequency MRI (WMCF)", "Low-Frequency PET (WMCF)"])

# Structure Tensor Fusion
FH = apply_structure_tensor_and_energy(AH, BH)
FL = low_frequency_fusion(AL, BL)

# XDoG Edge Detection
XDOG_A = xdog(A)
XDOG_B = xdog(B)
# COA Optimization Dummy Data (for plotting purposes)
costs = np.random.rand(20)
plot_coa(None, costs)


# Plot Intermediate Results
plot_images(
    [A, B, AH, BH, FH, FL, XDOG_A, XDOG_B],
    ["MRI Image", "PET Image", "High-Frequency MRI", "High-Frequency PET", "Fused HF", "Fused LF", "XDoG MRI", "XDoG PET"]
)


