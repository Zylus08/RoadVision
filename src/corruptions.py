"""Environmental corruption pipeline for autonomous driving robustness evaluation.

Implements realistic atmospheric and sensor corruptions:
1. Low Light: Non-linear intensity compression and gamma attenuation.
2. Fog: Atmospheric scattering model with depth approximation.
3. Rain: Synthetic directional rain streaks with contrast reduction.
4. Gaussian Blur: Optical defocus and motion-like Gaussian filtering.
5. Gaussian Noise: Sensor shot noise with configurable variance.

All corruptions support severity levels 1 through 5 and deterministic random seeds.
"""

import numpy as np
import cv2

from src.config import get_corruption_config

def set_seed(seed: int = 42):
    """Set numpy random seed for deterministic corruptions."""
    np.random.seed(seed)

def corrupt_low_light(image: np.ndarray, severity: int = 1, seed: int = None) -> np.ndarray:
    """Simulate low-light / nighttime illumination conditions.

    Transformation:
        I' = alpha * (I / 255.0) ^ gamma * 255.0
    where alpha controls overall attenuation and gamma controls non-linear shadow compression.

    Args:
        image: RGB uint8 array (H, W, 3).
        severity: Level 1 (dusk) to 5 (pitch dark road).
        seed: Random seed (unused for purely deterministic function, kept for uniform API).

    Returns:
        Corrupted RGB image uint8 (H, W, 3).
    """
    severity = max(1, min(5, severity))
    corruption_config = get_corruption_config()
    low_light_config = corruption_config.get("low_light", {})
    params = low_light_config.get(severity, {"alpha": 0.50, "gamma": 2.0})

    img_float = image.astype(np.float32) / 255.0
    corrupted = params["alpha"] * np.power(img_float, params["gamma"]) * 255.0
    return np.clip(corrupted, 0, 255).astype(np.uint8)

def corrupt_fog(image: np.ndarray, severity: int = 1, seed: int = None) -> np.ndarray:
    """Simulate atmospheric fog using Koschmieder's atmospheric scattering model.

    Formulation:
        I_fog(x) = I(x) * t(x) + L * (1 - t(x))
    where:
        t(x) = exp(-beta * d(x)) is the atmospheric transmission,
        d(x) is the approximated road depth map (deeper at top/horizon),
        L is the atmospheric airlight vector (~0.85).

    Args:
        image: RGB uint8 array (H, W, 3).
        severity: Level 1 (light haze) to 5 (dense fog).
        seed: Random seed.

    Returns:
        Corrupted RGB image uint8 (H, W, 3).
    """
    severity = max(1, min(5, severity))
    corruption_config = get_corruption_config()
    fog_config = corruption_config.get("fog", {})
    beta = fog_config.get(severity, {}).get("beta", 2.0)
    airlight = fog_config.get(severity, {}).get("airlight", 220.0)

    H, W, _ = image.shape
    # Approximate depth gradient: top (horizon) is farther (d ~ 1.0), bottom (hood) is close (d ~ 0.1)
    y_coords = np.linspace(1.0, 0.15, H).reshape(H, 1)
    depth_map = np.repeat(y_coords, W, axis=1)[:, :, np.newaxis]

    # Atmospheric transmission
    transmission = np.exp(-beta * depth_map)

    img_float = image.astype(np.float32)
    foggy = img_float * transmission + airlight * (1.0 - transmission)
    return np.clip(foggy, 0, 255).astype(np.uint8)

def corrupt_rain(image: np.ndarray, severity: int = 1, seed: int = None) -> np.ndarray:
    """Simulate synthetic rain streaks and road water film contrast attenuation.

    Args:
        image: RGB uint8 array (H, W, 3).
        severity: Level 1 (light drizzle) to 5 (torrential downpour).
        seed: Random seed for deterministic streak pattern.

    Returns:
        Corrupted RGB image uint8 (H, W, 3).
    """
    # Use Generator for thread-safe random operations
    rng = np.random.default_rng(seed)

    severity = max(1, min(5, severity))
    corruption_config = get_corruption_config()
    rain_config = corruption_config.get("rain", {})
    configs = rain_config.get(severity, {
        "drops": 2600, "length": 30, "angle": -35, "blur": 5, "darken": 0.82
    })

    H, W, C = image.shape
    streak_mask = np.zeros((H, W), dtype=np.float32)

    # Generate random streak centers
    num_drops = configs["drops"]
    x = rng.integers(0, W, num_drops)
    y = rng.integers(0, H, num_drops)
    intensity = rng.uniform(150, 255, num_drops)

    # Vectorized scatter using np.add.at (handles duplicates correctly)
    np.add.at(streak_mask, (y, x), intensity)

    # Create directional motion blur kernel for rain streak elongation
    length = configs["length"]
    angle = configs["angle"]
    rad = np.deg2rad(angle)
    kernel = np.zeros((length, length), dtype=np.float32)
    cx, cy = length // 2, length // 2
    dx = int(cx * np.cos(rad))
    dy = int(cy * np.sin(rad))
    cv2.line(kernel, (cx - dx, cy - dy), (cx + dx, cy + dy), 1.0, thickness=1)
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum

    streak_mask = cv2.filter2D(streak_mask, -1, kernel)
    streak_mask = cv2.GaussianBlur(streak_mask, (configs["blur"], configs["blur"]), 0)

    # Blend streaks with original image and attenuate contrast
    darkened = image.astype(np.float32) * configs["darken"]
    rainy = darkened + streak_mask[:, :, np.newaxis] * 0.7
    return np.clip(rainy, 0, 255).astype(np.uint8)

def corrupt_gaussian_blur(image: np.ndarray, severity: int = 1, seed: int = None) -> np.ndarray:
    """Simulate optical defocus, dirty sensor lens, or high-speed motion blur.

    Args:
        image: RGB uint8 array (H, W, 3).
        severity: Level 1 (mild blur) to 5 (heavy blur).
        seed: Random seed.

    Returns:
        Corrupted RGB image uint8 (H, W, 3).
    """
    severity = max(1, min(5, severity))
    corruption_config = get_corruption_config()
    blur_config = corruption_config.get("gaussian_blur", {})
    params = blur_config.get(severity, {"ksize": 15, "sigma": 3.8})

    ksize = params["ksize"]
    sigma = params["sigma"]
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    return blurred

def corrupt_gaussian_noise(image: np.ndarray, severity: int = 1, seed: int = None) -> np.ndarray:
    """Simulate camera sensor electronic shot noise and ISO gain artifacts.

    Args:
        image: RGB uint8 array (H, W, 3).
        severity: Level 1 (subtle grain) to 5 (severe noise).
        seed: Random seed.

    Returns:
        Corrupted RGB image uint8 (H, W, 3).
    """
    # Use Generator for thread-safe random operations
    rng = np.random.default_rng(seed)

    severity = max(1, min(5, severity))
    corruption_config = get_corruption_config()
    noise_config = corruption_config.get("gaussian_noise", {})
    sigma = noise_config.get(severity, {}).get("sigma", 32.0)

    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

CORRUPTION_REGISTRY = {
    "low_light": corrupt_low_light,
    "fog": corrupt_fog,
    "rain": corrupt_rain,
    "gaussian_blur": corrupt_gaussian_blur,
    "gaussian_noise": corrupt_gaussian_noise
}

def apply_corruption(image: np.ndarray, corruption_type: str, severity: int = 1, seed: int = 42) -> np.ndarray:
    """Universal dispatcher for environmental corruptions.
    
    Args:
        image: Original RGB uint8 array (H, W, 3).
        corruption_type: One of 'low_light', 'fog', 'rain', 'gaussian_blur', 'gaussian_noise'.
        severity: Integer from 1 to 5.
        seed: Random seed for deterministic reproducibility.
        
    Returns:
        Corrupted RGB uint8 array (H, W, 3).
    """
    if corruption_type not in CORRUPTION_REGISTRY:
        raise ValueError(f"Unknown corruption type '{corruption_type}'. Supported: {list(CORRUPTION_REGISTRY.keys())}")
        
    fn = CORRUPTION_REGISTRY[corruption_type]
    return fn(image, severity=severity, seed=seed)
