# Image Frequency Spectrum Analysis UI

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Interactive Streamlit application for image frequency spectrum analysis and quantitative comparison of multiple images using 2D FFT (Fast Fourier Transform)

## ✨ Key Features

### 🔬 Frequency Analysis
- **FFT Magnitude Spectrum**: Visualize frequency components in Log10 scale
- **RGB Channel Analysis**: Analyze R, G, B channels separately

### 📊 Radius Profile
- **Log Power vs Radius**: Quantify power distribution by distance from center
- **Statistics**: Mean, median, percentiles (25-75%)

### 🔍 Multiple Image Comparison
- **Side-by-Side Display**: Compare spectra of multiple images
- **Overlay Plots**: Quantitatively compare radius profiles

## 🚀 Quick Start

### Requirements

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) (Python package manager)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/kroobose/freq-spectrum-viewer.git
cd freq-spectrum-viewer
```

2. **Install uv** (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. **Install dependencies**

```bash
uv sync
```

### Launch

```bash
uv run streamlit run app.py
```

The browser will automatically open and display the application (default: `http://localhost:8501`)

## 📖 Usage

### Basic Workflow

1. **Upload Images**
   - Select one or more images from the left sidebar
   - Supported formats: PNG, JPG, JPEG, BMP

2. **Configure Display Options**
   - Toggle color/grayscale display
   - Enable/disable FFT spectrum, radius profile
   - Select colormap
   - Enable RGB channel analysis (optional)

3. **Customize Radius Profile** (Optional)
   - Choose Y-axis display mode (Log Power / Linear Power / Magnitude)
   - Toggle log scale for radius axis
   - Show percentile ranges

4. **View Results**
   - Visual spectrum comparison
   - Quantitative comparison using radius profiles
   - Detailed statistics

## 🛠 Technical Details

### Architecture

```
app.py                # Streamlit UI & visualization
freq_analysis.py      # FFT analysis core logic
```

### FFT Analysis Pipeline

1. **Image Loading**: Preserve both color (RGB) and grayscale (L) versions
2. **FFT Computation**: Execute 2D FFT on grayscale image
3. **FFT Shift**: Center DC component (zero frequency)
4. **Magnitude**: Calculate absolute value of complex numbers
5. **Log Transform**: Normalize with Log10 scale

### Radius Profile Calculation

1. Calculate distance from center for each pixel
2. Group pixels by distance (binning)
3. Compute statistics at each distance (mean, median, percentiles)
4. Transform to Log/Linear based on user selection

## 📊 Theoretical Background

### FFT Magnitude Spectrum

2D FFT transforms images from spatial domain to frequency domain:

- **Center (DC component)**: Average image brightness
- **Low frequency components**: Overall image structure
- **High frequency components**: Fine details like edges and textures

### Radius Profile

Aggregating power by distance (radius) from center enables **rotation-invariant** quantification of frequency distribution. This is useful for image quality assessment and comparison.

## 📁 Project Structure

```
freq-spectrum-viewer/
├── app.py              # Streamlit main application
├── freq_analysis.py    # FFT analysis core module
├── pyproject.toml      # Project settings and dependencies
├── .python-version     # Python version specification
└── README.md           # This file
```

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [NumPy](https://numpy.org/) - Fast numerical computation
- [Matplotlib](https://matplotlib.org/) - Beautiful visualizations
- [Streamlit](https://streamlit.io/) - Easy web app development
- [Pillow](https://python-pillow.org/) - Image processing

⭐ If you found this project helpful, please give it a star!
