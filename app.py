"""
Image Frequency Spectrum Analysis UI

Interactive Streamlit UI for FFT spectrum analysis and comparison of multiple images.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import freq_analysis as fa

# Page configuration
st.set_page_config(
    page_title="Frequency Spectrum Analysis",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Image Frequency Spectrum Analysis")
st.markdown("Upload multiple images to compare FFT spectra and radius profiles")

# Sidebar settings
st.sidebar.header("⚙️ Settings")

# Image upload
uploaded_files = st.sidebar.file_uploader(
    "Select images",
    type=["png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True,
    help="Select multiple images to compare"
)

# Display options
st.sidebar.subheader("Display Options")
show_color = st.sidebar.checkbox("Show color images", value=False, help="Show grayscale if unchecked")
show_original = st.sidebar.checkbox("Show original images", value=True)
show_spectrum = st.sidebar.checkbox("Show FFT spectrum", value=True)

# RGB channel analysis
analyze_rgb_channels = st.sidebar.checkbox("Analyze RGB channels separately", value=False, help="Compute FFT for each RGB channel")

show_profile = st.sidebar.checkbox("Show radius profile", value=False)

# Colormap selection
cmap_options = ['viridis', 'plasma', 'inferno', 'magma', 'jet', 'hot', 'coolwarm']
colormap = st.sidebar.selectbox(
    "Colormap",
    options=cmap_options,
    index=0,
    help="Select colormap for spectrum display"
)

# Profile display options
if show_profile:
    st.sidebar.subheader("Radius Profile Settings")

    # Y-axis display mode
    power_mode = st.sidebar.radio(
        "Y-axis Display Mode",
        options=["Log Power", "Linear Power", "Magnitude"],
        index=0,
        help="Log Power: log10 scale / Linear Power: 10^x transformed / Magnitude: amplitude"
    )

    # X-axis display mode
    log_radius = st.sidebar.checkbox("Log scale for radius axis", value=False)

    # Statistics display
    show_percentiles = st.sidebar.checkbox("Show percentiles", value=False)
else:
    power_mode = "Log Power"
    log_radius = False
    show_percentiles = False


def plot_spectrum(spectrum, title, colormap='viridis', show_colorbar=True):
    """Plot FFT spectrum with correct frequency axes"""
    fig, ax = plt.subplots(figsize=(6, 6))

    h, w = spectrum.shape

    # Create extent for correct frequency labeling
    # After fftshift, center is DC (0 frequency)
    # Frequency ranges from -0.5 to 0.5 (normalized frequency)
    extent = [-0.5, 0.5, 0.5, -0.5]  # [left, right, bottom, top]

    im = ax.imshow(spectrum, cmap=colormap, interpolation='nearest',
                   extent=extent, origin='upper')
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('Normalized Frequency X', fontsize=10)
    ax.set_ylabel('Normalized Frequency Y', fontsize=10)

    # Add grid lines at zero frequency
    ax.axhline(y=0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)

    # Add text annotation for DC component
    ax.text(0.02, 0.98, 'DC (0,0)', transform=ax.transAxes,
            color='white', fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    if show_colorbar:
        plt.colorbar(im, ax=ax, label='Log10 Magnitude')
    plt.tight_layout()
    return fig


def plot_radius_profile(radii, power, stats=None, title="Radius Profile",
                        log_radius=False, show_percentiles=False, power_mode="Log Power"):
    """
    Plot radius profile

    Args:
        power_mode: One of "Log Power", "Linear Power", "Magnitude"
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Transform data
    if power_mode == "Linear Power":
        # Convert from log10 scale (10^x)
        display_power = np.power(10, power)
        ylabel = 'Power'
        if stats and 'median' in stats:
            display_median = np.power(10, stats['median'])
        if show_percentiles and stats and 'percentiles' in stats:
            p25 = np.power(10, stats['percentiles'][25]) if 25 in stats['percentiles'] else None
            p75 = np.power(10, stats['percentiles'][75]) if 75 in stats['percentiles'] else None
    elif power_mode == "Magnitude":
        # Convert from log10 scale to magnitude
        display_power = np.power(10, power)
        ylabel = 'Magnitude'
        if stats and 'median' in stats:
            display_median = np.power(10, stats['median'])
        if show_percentiles and stats and 'percentiles' in stats:
            p25 = np.power(10, stats['percentiles'][25]) if 25 in stats['percentiles'] else None
            p75 = np.power(10, stats['percentiles'][75]) if 75 in stats['percentiles'] else None
    else:  # Log Power
        display_power = power
        ylabel = 'Log Power'
        if stats and 'median' in stats:
            display_median = stats['median']
        if show_percentiles and stats and 'percentiles' in stats:
            p25 = stats['percentiles'][25] if 25 in stats['percentiles'] else None
            p75 = stats['percentiles'][75] if 75 in stats['percentiles'] else None

    # Main power profile
    ax.plot(radii, display_power, 'b-', linewidth=2, label='Mean (average)')

    # Median
    if stats and 'median' in stats:
        ax.plot(radii, display_median, 'g--', linewidth=1.5, label='Median (50th percentile)', alpha=0.7)

    # Percentiles
    if show_percentiles and stats and 'percentiles' in stats:
        if p25 is not None and p75 is not None:
            ax.fill_between(
                radii,
                p25,
                p75,
                alpha=0.3,
                color='blue',
                label='25-75 Percentile'
            )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('Radius (pixels)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if log_radius:
        ax.set_xscale('log')

    # For Linear Power or Magnitude, log scale on Y-axis is more readable
    if power_mode in ["Linear Power", "Magnitude"]:
        ax.set_yscale('log')

    plt.tight_layout()
    return fig


# Main processing
if uploaded_files:
    st.sidebar.success(f"✅ Loaded {len(uploaded_files)} image(s)")

    # Analyze images
    results = []
    for uploaded_file in uploaded_files:
        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            result = fa.analyze_image(uploaded_file, compute_profile=show_profile, analyze_rgb=analyze_rgb_channels)
            result['name'] = uploaded_file.name
            results.append(result)

    # Display layout
    num_images = len(results)

    if show_original:
        st.header("📷 Original Images")
        display_mode = "Color" if show_color else "Grayscale"
        st.caption(f"Display mode: {display_mode}")

        cols = st.columns(min(num_images, 3))
        for i, result in enumerate(results):
            with cols[i % 3]:
                # Toggle color/grayscale
                if show_color:
                    # Display color image
                    pil_img = Image.fromarray(result['original_color'], mode='RGB')
                else:
                    # Display grayscale image
                    img_uint8 = np.clip(result['original_gray'], 0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_uint8, mode='L')
                st.image(pil_img, caption=result['name'])

    if show_spectrum:
        st.header("🌈 FFT Magnitude Spectrum (Log Scale)")

        # RGBチャンネル解析が有効な場合
        if analyze_rgb_channels and any('rgb_spectra' in r for r in results):
            for result in results:
                st.subheader(f"{result['name']}")

                # 4列で表示：R, G, B, Combined
                col1, col2, col3, col4 = st.columns(4)

                if 'rgb_spectra' in result:
                    rgb_sp = result['rgb_spectra']

                    with col1:
                        fig = plot_spectrum(rgb_sp['R'], "Red Channel", colormap='Reds', show_colorbar=False)
                        st.pyplot(fig)
                        plt.close(fig)

                    with col2:
                        fig = plot_spectrum(rgb_sp['G'], "Green Channel", colormap='Greens', show_colorbar=False)
                        st.pyplot(fig)
                        plt.close(fig)

                    with col3:
                        fig = plot_spectrum(rgb_sp['B'], "Blue Channel", colormap='Blues', show_colorbar=False)
                        st.pyplot(fig)
                        plt.close(fig)

                    with col4:
                        fig = plot_spectrum(rgb_sp['combined'], "Combined (Average)", colormap=colormap, show_colorbar=True)
                        st.pyplot(fig)
                        plt.close(fig)
        else:
            # 通常のグレースケールスペクトル表示
            cols = st.columns(min(num_images, 3))
            for i, result in enumerate(results):
                with cols[i % 3]:
                    fig = plot_spectrum(
                        result['spectrum'],
                        f"{result['name']}\n({result['shape'][0]}x{result['shape'][1]} pixels)",
                        colormap=colormap
                    )
                    st.pyplot(fig)
                    plt.close(fig)



    if show_profile and all('radii' in r for r in results):
        st.header("📈 Radius Profile (Log Power vs Radius)")

        # Individual plots
        if len(results) == 1:
            result = results[0]
            fig = plot_radius_profile(
                result['radii'],
                result['power'],
                result.get('stats'),
                title=f"Radius Profile - {result['name']}",
                log_radius=log_radius,
                show_percentiles=show_percentiles,
                power_mode=power_mode
            )
            st.pyplot(fig)
            plt.close(fig)

        # Multiple image comparison plots
        else:
            # Individual plots
            st.subheader("Individual Profiles")
            cols = st.columns(min(num_images, 2))
            for i, result in enumerate(results):
                with cols[i % 2]:
                    fig = plot_radius_profile(
                        result['radii'],
                        result['power'],
                        result.get('stats'),
                        title=result['name'],
                        log_radius=log_radius,
                        show_percentiles=show_percentiles,
                        power_mode=power_mode
                    )
                    st.pyplot(fig)
                    plt.close(fig)

            # Overlay plot
            st.subheader("Comparison Plot (Overlay)")
            fig, ax = plt.subplots(figsize=(10, 6))

            colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
            for i, result in enumerate(results):
                ax.plot(
                    result['radii'],
                    result['power'],
                    color=colors[i],
                    linewidth=2,
                    label=result['name'],
                    alpha=0.8
                )

            ax.set_title("All Images - Radius Profile Comparison", fontsize=14, pad=10)
            ax.set_xlabel('Radius (pixels)', fontsize=11)
            ax.set_ylabel('Log Power', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            if log_radius:
                ax.set_xscale('log')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Statistics
    with st.expander("📊 Detailed Statistics"):
        for result in results:
            st.subheader(result['name'])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Image Size", f"{result['shape'][1]} x {result['shape'][0]}")
            with col2:
                st.metric("Spectrum Max", f"{np.max(result['spectrum']):.2f}")
            with col3:
                st.metric("Spectrum Min", f"{np.min(result['spectrum']):.2f}")

            if 'radii' in result:
                st.write(f"**Radius Range:** 0 - {np.max(result['radii']):.0f} pixels")
                st.write(f"**Power Range:** {np.min(result['power']):.2f} - {np.max(result['power']):.2f}")

else:
    # Instructions
    st.info("👈 Upload images from the sidebar to get started")

    st.markdown("""
    ## How to Use

    1. **Upload Images**: Select one or more images from the sidebar
    2. **Choose Display Options**:
       - Show/hide original images
       - Show/hide FFT spectrum
       - Show/hide radius profile
    3. **Select Colormap**: Customize the spectrum color scheme
    4. **View Results**:
       - Compare multiple images side by side
       - Quantitatively compare using radius profiles

    ## Features

    ### FFT Magnitude Spectrum
    Visualizes the frequency components of the image. The center represents low frequencies, and the outer regions represent high frequencies.
    - **Bright areas**: Strong frequency components
    - **Dark areas**: Weak frequency components

    ### Radius Profile
    A graph aggregating power at each distance (radius) from the center.
    - **X-axis**: Distance from center (pixels)
    - **Y-axis**: Log Power (power at that distance)
    - **Mean (blue line)**: Average power across all pixels at each radius
    - **Median (green dashed line)**: 50th percentile (middle value) of power at each radius
    - Enables quantitative comparison of image frequency characteristics
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
This application performs frequency analysis on images.
- **Technology**: Python, NumPy, Matplotlib, Streamlit
- **Analysis Method**: 2D FFT (Fast Fourier Transform)
""")
