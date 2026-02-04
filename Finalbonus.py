# Algorithm 1: XDoG_ST_LE (High-frequency synthesis)
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, sobel

# This function calculates the Structure Tensor of an image, which provides local edge orientation and intensity information.
def structure_tensor(image, sigma=1.0):
    """
    Computes the Structure Tensor of an image.
    :param image: Grayscale image (n x m single-channel matrix).
    :param sigma: Gaussian smoothing factor.
    :return: Structure Tensor components (Jxx, Jyy, Jxy).
    """
    Ix = sobel(image, axis=1, mode='reflect')  # Derivative along the x-axis.
    Iy = sobel(image, axis=0, mode='reflect')  # Derivative along the y-axis.
    Ix2 = gaussian_filter(Ix**2, sigma)       # Smoothed squared derivative along x.
    Iy2 = gaussian_filter(Iy**2, sigma)       # Smoothed squared derivative along y.
    Ixy = gaussian_filter(Ix * Iy, sigma)     # Smoothed product of derivatives.
    return Ix2, Iy2, Ixy

# Computes the Local Energy Function based on Structure Tensor eigenvalues, capturing texture and feature saliency.
def local_energy(Ix2, Iy2, Ixy):
    """
    Computes the Local Energy Function from Structure Tensor components.
    :return: Local energy image.
    """
    trace = Ix2 + Iy2
    determinant = (Ix2 * Iy2) - (Ixy**2)
    energy = trace - np.sqrt(np.maximum(trace**2 - 4 * determinant, 0))
    return energy

# This function implements the XDoG filter, an advanced edge detection method preserving edges while reducing noise.
def xdog(image, epsilon=0.01, k=200, gamma=0.98, phi=10):
    """
    Computes the eXtended Difference of Gaussians (XDoG).
    :param image: Grayscale image.
    """
    s1, s2 = 0.5, 0.5 * k
    gauss1 = gaussian_filter(image, s1)          # First Gaussian smoothing.
    gauss2 = gamma * gaussian_filter(image, s2) # Second Gaussian smoothing with scaling.
    dog_result = gauss1 - gauss2                # Difference of Gaussians.
    dog_result = dog_result / 255.0            # Normalize the result.
    mask = dog_result >= epsilon
    dog_result[mask] = 1.0                      # Thresholding for edge emphasis.
    dog_result[~mask] = 1.0 + np.tanh(phi * (dog_result[~mask] - epsilon))
    return (dog_result * 255).astype(np.uint8)

# Fuses high-frequency components from two images using Structure Tensor and Local Energy functions.
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

# Algorithm 2: ASR_COA (Low-frequency synthesis)
def COA_ASR(AL, BL, AH, FH, CF, nfevalMAX, n_packs=20, n_coy=5):
    """
    Adaptive Synthesis Rule using Coyote Optimization Algorithm (ASR_COA).
    """
    VarMin = np.array([0.9, 0.001])  # Lower bounds for parameters.
    VarMax = np.array([0.999, 0.3])  # Upper bounds for parameters.
    D = len(VarMin)
    if n_coy < 3:
        raise Exception("At least 3 coyotes per pack must be used")
    pop_total = n_packs * n_coy
    coyotes = VarMin + np.random.rand(pop_total, D) * (VarMax - VarMin)  # Initialize population.
    costs = np.zeros(pop_total)
    for i in range(pop_total):
        FL = coyotes[i, 0] * AL + coyotes[i, 1] * BL  # Low-frequency fusion.
        F = FL + FH                                   # Combine with high-frequency components.
        costs[i] = CF(F)                             # Evaluate fitness.
    nfeval = pop_total
    globalMin = np.min(costs)
    globalParams = coyotes[np.argmin(costs), :]
    while nfeval < nfevalMAX:
        for p in range(n_packs):
            pack_indices = np.arange(p * n_coy, (p + 1) * n_coy)
            coyotes_aux = coyotes[pack_indices, :]
            costs_aux = costs[pack_indices]
            ind = np.argsort(costs_aux)
            costs_aux = costs_aux[ind]
            coyotes_aux = coyotes_aux[ind, :]
            c_alpha = coyotes_aux[0, :]                  # Best coyote in the pack.
            tendency = np.median(coyotes_aux, axis=0)   # Social tendency.
            new_coyotes = np.zeros_like(coyotes_aux)
            for c in range(n_coy):
                rc1, rc2 = np.random.choice(n_coy, 2, replace=False)
                new_coyotes[c, :] = (
                    coyotes_aux[c, :] + np.random.rand() * (c_alpha - coyotes_aux[rc1, :])
                    + np.random.rand() * (tendency - coyotes_aux[rc2, :])
                )
                new_coyotes[c, :] = np.clip(new_coyotes[c, :], VarMin, VarMax)
                FL = new_coyotes[c, 0] * AL + new_coyotes[c, 1] * BL
                F = FL + FH
                new_cost = CF(F)
                nfeval += 1
                if new_cost < costs_aux[c]:
                    costs_aux[c] = new_cost
                    coyotes_aux[c, :] = new_coyotes[c, :]
            coyotes[pack_indices, :] = coyotes_aux
            costs[pack_indices] = costs_aux
        globalMin = np.min(costs)
        globalParams = coyotes[np.argmin(costs), :]
    FL = globalParams[0] * AL.astype('float32') + globalParams[1] * BL.astype('float32')
    return FL, globalParams

# Algorithm 3: IEXDoG_COA (Fusion)
def fuse_images(IMRI, IPET, nfevalMAX=1000):
    """
    Fuses MRI and PET images using IEXDoG_COA.
    """
    A = cv2.convertScaleAbs(IMRI, alpha=1.2, beta=30).astype('float32')  # Brightness adjustment.
    PET_YUV = cv2.cvtColor(IPET, cv2.COLOR_BGR2YUV)
    B = PET_YUV[:, :, 0].astype('float32')  # Extract luminance.
    U, V = PET_YUV[:, :, 1], PET_YUV[:, :, 2]  # Extract chrominance.
    AL = gaussian_filter(A, 5).astype('float32')  # Low-frequency MRI.
    AH = A - AL                                 # High-frequency MRI.
    BL = gaussian_filter(B, 5).astype('float32')  # Low-frequency PET.
    BH = B - BL                                 # High-frequency PET.
    FH = apply_structure_tensor_and_energy(AH, BH).astype('float32')
    FL, _ = COA_ASR(AL, BL, AH, FH, lambda F: np.std(F), nfevalMAX)  # Optimize low-frequency fusion.
    FL = FL.astype('float32')
    FH = FH.astype('float32')
    F = cv2.addWeighted(FL, 0.5, FH, 0.5, 0).astype('float32')  # Combine low and high frequencies.
    F_YUV = np.zeros_like(PET_YUV, dtype=np.uint8)
    F_YUV[:, :, 0] = cv2.normalize(F, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Normalize luminance.
    F_YUV[:, :, 1] = U  # Preserve chrominance.
    F_YUV[:, :, 2] = V
    return cv2.cvtColor(F_YUV, cv2.COLOR_YUV2BGR)  # Convert back to RGB.

if __name__ == '__main__':
    IMRI = cv2.imread('MRI1.png', cv2.IMREAD_GRAYSCALE).astype('float32')  # Read MRI image.
    IPET = cv2.imread('PET.png', cv2.IMREAD_COLOR)                        # Read PET image.
    if IMRI is None or IPET is None:
        raise FileNotFoundError("One or both input images not found!")
    fused_image = fuse_images(IMRI, IPET)
    cv2.imwrite('Fused_Image.jpg', fused_image)  # Save fused image.
    print("Fused image saved as 'Fused_Image.jpg'")
