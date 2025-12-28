"""
Image frequency spectrum analysis module

This module provides:
- FFT magnitude spectrum calculation
- Radius profile calculation (power profile based on distance from center)
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional


def load_and_convert_image(image_file) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load image and return both color and grayscale versions

    Args:
        image_file: PIL Image or file path

    Returns:
        (color image numpy array, grayscale image numpy array)
    """
    if isinstance(image_file, str):
        img = Image.open(image_file)
    else:
        img = Image.open(image_file)

    # Keep original color image (convert to RGB)
    if img.mode != 'RGB':
        img_color = img.convert('RGB')
    else:
        img_color = img.copy()

    color_array = np.array(img_color, dtype=np.uint8)

    # Convert to grayscale
    img_gray = img.convert('L')
    gray_array = np.array(img_gray, dtype=np.float64)

    return color_array, gray_array


def compute_fft_spectrum(image: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """
    Compute FFT magnitude spectrum of an image

    Args:
        image: Input image (grayscale)
        log_scale: If True, return in log10 scale

    Returns:
        FFT magnitude spectrum (shifted)
    """
    # Compute 2D FFT
    fft = np.fft.fft2(image)

    # Shift FFT to center DC component
    fft_shifted = np.fft.fftshift(fft)

    # Calculate magnitude
    magnitude = np.abs(fft_shifted)

    # Normalize with log scale
    if log_scale:
        # Add small value to avoid division by zero
        magnitude = np.log10(magnitude + 1e-10)

    return magnitude


def compute_radius_profile(
    spectrum: np.ndarray,
    percentiles: Optional[list] = None,
    use_log: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute power profile based on distance from center

    Args:
        spectrum: FFT magnitude spectrum
        percentiles: List of percentile values to compute (e.g., [25, 50, 75])
        use_log: Assume spectrum is in log scale

    Returns:
        radii: Array of unique radius values (sorted)
        mean_power: Mean power at each radius
        stats: Additional statistics (median, percentiles, etc.)
    """
    h, w = spectrum.shape
    center_y, center_x = h // 2, w // 2

    # Calculate distance from center for each pixel
    y, x = np.ogrid[:h, :w]
    radius_map = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Round radius to integers (binning)
    radius_int = np.round(radius_map).astype(int)

    # Get unique radius values
    unique_radii = np.unique(radius_int)

    # Calculate statistics at each radius
    mean_power = np.zeros(len(unique_radii))
    stats = {'use_log': use_log}

    if percentiles:
        stats['percentiles'] = {p: np.zeros(len(unique_radii)) for p in percentiles}

    for i, r in enumerate(unique_radii):
        mask = (radius_int == r)
        values = spectrum[mask]

        mean_power[i] = np.mean(values)

        if percentiles:
            for p in percentiles:
                stats['percentiles'][p][i] = np.percentile(values, p)

    stats['median'] = np.array([np.median(spectrum[radius_int == r]) for r in unique_radii])

    return unique_radii, mean_power, stats


def compute_rgb_spectra(color_image: np.ndarray, log_scale: bool = True) -> dict:
    """
    Compute FFT magnitude spectrum for each RGB channel

    Args:
        color_image: Input image (RGB)
        log_scale: If True, return in log10 scale

    Returns:
        Dictionary containing spectrum for each channel:
        {
            'R': R channel spectrum,
            'G': G channel spectrum,
            'B': B channel spectrum,
            'combined': RGB combined spectrum (average)
        }
    """
    # Separate channels
    r_channel = color_image[:, :, 0].astype(np.float64)
    g_channel = color_image[:, :, 1].astype(np.float64)
    b_channel = color_image[:, :, 2].astype(np.float64)

    # Compute spectrum for each channel
    r_spectrum = compute_fft_spectrum(r_channel, log_scale=log_scale)
    g_spectrum = compute_fft_spectrum(g_channel, log_scale=log_scale)
    b_spectrum = compute_fft_spectrum(b_channel, log_scale=log_scale)

    # Combined spectrum (average)
    if log_scale:
        # In log space, take linear average
        combined_spectrum = (r_spectrum + g_spectrum + b_spectrum) / 3.0
    else:
        combined_spectrum = (r_spectrum + g_spectrum + b_spectrum) / 3.0

    return {
        'R': r_spectrum,
        'G': g_spectrum,
        'B': b_spectrum,
        'combined': combined_spectrum
    }


def analyze_image(
    image_file,
    compute_profile: bool = True,
    analyze_rgb: bool = False
) -> dict:
    """
    Perform complete frequency analysis of an image

    Args:
        image_file: Image file
        compute_profile: Whether to compute radius profile
        analyze_rgb: Whether to analyze RGB channels separately

    Returns:
        Dictionary containing analysis results:
        {
            'original_color': Original image (color),
            'original_gray': Original image (grayscale),
            'spectrum': FFT magnitude spectrum,
            'rgb_spectra': RGB channel spectra (if analyze_rgb=True),
            'radii': Radius array (if profile computed),
            'power': Power array (if profile computed),
            'stats': Statistics
        }
    """
    # Load image (color and grayscale)
    img_color, img_gray = load_and_convert_image(image_file)

    # Compute FFT spectrum (using grayscale image)
    spectrum = compute_fft_spectrum(img_gray, log_scale=True)

    result = {
        'original_color': img_color,
        'original_gray': img_gray,
        'spectrum': spectrum,
        'shape': img_gray.shape
    }

    # RGB analysis
    if analyze_rgb:
        rgb_spectra = compute_rgb_spectra(img_color, log_scale=True)
        result['rgb_spectra'] = rgb_spectra

    # Compute radius profile
    if compute_profile:
        radii, power, stats = compute_radius_profile(spectrum, percentiles=[25, 75], use_log=True)
        result['radii'] = radii
        result['power'] = power
        result['stats'] = stats

    return result
