"""
Neural Frequency Fusion Studio (NFFS)
======================================
Adaptive spatial–frequency domain image transformation
with patch-level neural decision making.

Dependencies:
    pip install numpy opencv-python matplotlib scipy

Usage:
    python nffs.py                          # uses built-in synthetic image
    python nffs.py --image path/to/img.png  # use your own image
    python nffs.py --noise 20 --blur 2.0    # custom degradation
"""

import argparse
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


# ─────────────────────────────────────────────
# 1. IMAGE GENERATION / LOADING
# ─────────────────────────────────────────────

def generate_synthetic_image(size=256, seed=42):
    """
    Create a synthetic test image with mixed textures:
    smooth gradients + high-frequency details + edges.
    Simulates a natural image for DIP experiments.
    """
    np.random.seed(seed)
    W = H = size
    x = np.linspace(0, 2 * np.pi, W)
    y = np.linspace(0, 2 * np.pi, H)
    X, Y = np.meshgrid(x, y)

    # R channel: low-frequency smooth base + mid-frequency texture
    R = 128 + 40 * np.sin(X / 1.2) + 30 * np.cos(Y / 0.9) \
        + 20 * np.sin((X + Y) / 1.5) + 15 * np.sin(X / 0.4)
    # G channel: slightly warmer
    G = R * 0.85 + 10 * np.sin(X * Y / 4)
    # B channel: cooler tone
    B = R * 0.7 + 20 * np.cos(X / 0.7)

    img = np.stack([
        np.clip(R, 0, 255),
        np.clip(G, 0, 255),
        np.clip(B, 0, 255)
    ], axis=2).astype(np.uint8)

    return img


def load_image(path, size=256):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img


# ─────────────────────────────────────────────
# 2. DEGRADATION ENGINE
# ─────────────────────────────────────────────

def degrade_image(img, noise_sigma=18, blur_radius=1.5, seed=42):
    """
    Apply synthetic degradation:
        - Gaussian noise (additive white noise model)
        - Gaussian blur (simulates lens blur / compression)
    """
    np.random.seed(seed)
    degraded = img.astype(np.float32) + np.random.normal(0, noise_sigma, img.shape)
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)
    # Apply per-channel blur
    degraded = np.stack(
        [gaussian_filter(degraded[:, :, c], blur_radius) for c in range(3)],
        axis=2
    )
    return degraded.astype(np.uint8)


# ─────────────────────────────────────────────
# 3. SPATIAL ENHANCEMENT
# ─────────────────────────────────────────────

def spatial_enhance(img):
    """
    Unsharp masking via Laplacian sharpening kernel.
    Emphasises edges and fine detail in spatial domain.
    Kernel: 3x3 centre-weighted Laplacian sharpener.
    """
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    enhanced = np.stack([
        cv2.filter2D(img[:, :, c].astype(np.float32), -1, kernel)
        for c in range(3)
    ], axis=2)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 4. FREQUENCY FILTERING
# ─────────────────────────────────────────────

def ideal_low_pass(channel, cutoff=0.35, smooth_factor=10.0):
    """
    Frequency-domain low-pass filter with smooth Gaussian rolloff
    to avoid Gibbs ringing at the cutoff boundary.

    Steps:
        1. FFT → shift DC to centre
        2. Apply distance-based soft mask
        3. Inverse FFT → reconstruct
    """
    f = np.fft.fft2(channel.astype(np.float32))
    fshift = np.fft.fftshift(f)

    H, W = channel.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[-cy:H - cy, -cx:W - cx]
    dist = np.sqrt((xx / W) ** 2 + (yy / H) ** 2)

    # Soft mask: pass frequencies below cutoff, suppress above
    mask = np.exp(-smooth_factor * np.maximum(0, dist - cutoff) ** 2)
    fshift *= mask

    back = np.fft.ifft2(np.fft.ifftshift(fshift))
    return np.abs(back)


def frequency_filter(img, cutoff=0.35):
    """Apply ideal_low_pass to each colour channel independently."""
    filtered = np.stack([
        ideal_low_pass(img[:, :, c], cutoff=cutoff)
        for c in range(3)
    ], axis=2)
    return np.clip(filtered, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 5. PATCH FEATURE EXTRACTION
# ─────────────────────────────────────────────

def patch_features(patch):
    """
    Extract local features for the decision network:
        - variance:      texture richness
        - edge_density:  spatial gradient strength
        - mean:          overall brightness
    """
    gray = 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]
    variance = float(np.var(gray))
    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    edge_density = float(grad_x + grad_y)
    mean = float(gray.mean())
    return {"variance": variance, "edge_density": edge_density, "mean": mean}


# ─────────────────────────────────────────────
# 6. DECISION NETWORK (lightweight ML model)
# ─────────────────────────────────────────────

def decision_network(features, noise_sigma, feedback_bias=0.0):
    """
    Simulated 2-layer neural network:
        Input:  patch features + noise level
        Output: (spatial_weight, freq_weight, mode)

    Learned heuristics:
        High edge density  → prefer spatial sharpening
        High noise level   → prefer frequency suppression
        Mixed              → hybrid fusion

    feedback_bias: float in [-2, 2]
        Shifts decision boundary based on user reinforcement feedback.
        Positive = prefer spatial, Negative = prefer frequency.
    """
    noise_score  = min(1.0, noise_sigma / 60.0)
    edge_score   = min(1.0, features["edge_density"] / 12.0)
    texture_score = min(1.0, features["variance"] / 800.0)

    # Weighted combination (mimics a linear layer)
    sw = edge_score * 0.6 + texture_score * 0.2 - noise_score * 0.3 + feedback_bias * 0.1
    fw = noise_score * 0.7 - edge_score * 0.2 + 0.15 - feedback_bias * 0.05

    sw = max(0.0, sw)
    fw = max(0.0, fw)
    total = sw + fw if (sw + fw) > 0 else 1.0
    sw /= total
    fw /= total

    if sw > 0.6:
        mode = "spatial"
        reason = "High edge density → spatial sharpening preferred"
    elif fw > 0.6:
        mode = "freq"
        reason = "High noise / low edges → frequency suppression preferred"
    else:
        mode = "hybrid"
        reason = "Mixed characteristics → adaptive hybrid fusion"

    return {"spatial_w": sw, "freq_w": fw, "mode": mode, "reason": reason}


# ─────────────────────────────────────────────
# 7. HYBRID FUSION ENGINE
# ─────────────────────────────────────────────

def hybrid_fusion(degraded, spatial_img, freq_img, noise_sigma,
                  patch_size=32, manual_alpha=None, feedback_bias=0.0):
    """
    Main fusion step.
    For each patch:
        1. Extract features from the degraded patch
        2. Run decision network → get per-patch alpha (spatial weight)
        3. Blend: output = alpha * spatial + (1-alpha) * freq

    manual_alpha: float [0,1] or None
        If set, overrides the network for all patches (manual slider mode).
    """
    H, W = degraded.shape[:2]
    rows = H // patch_size
    cols = W // patch_size

    alpha_map  = np.zeros((H, W), dtype=np.float32)
    dec_grid   = np.zeros((rows, cols), dtype=int)   # 0=spatial,1=hybrid,2=freq
    dec_info   = {}

    for pr in range(rows):
        for pc in range(cols):
            py = pr * patch_size
            px = pc * patch_size
            patch = degraded[py:py + patch_size, px:px + patch_size]
            feat  = patch_features(patch)
            dec   = decision_network(feat, noise_sigma, feedback_bias)

            alpha = manual_alpha if manual_alpha is not None else dec["spatial_w"]
            alpha_map[py:py + patch_size, px:px + patch_size] = alpha

            mode_int = {"spatial": 0, "hybrid": 1, "freq": 2}[dec["mode"]]
            dec_grid[pr, pc] = mode_int
            dec_info[(pr, pc)] = {"features": feat, "decision": dec, "alpha": alpha}

    # Pixel-wise blend
    output = (alpha_map[:, :, np.newaxis] * spatial_img.astype(np.float32) +
              (1 - alpha_map[:, :, np.newaxis]) * freq_img.astype(np.float32))
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output, alpha_map, dec_grid, dec_info


# ─────────────────────────────────────────────
# 8. METRICS
# ─────────────────────────────────────────────

def compute_psnr(orig, processed):
    """Peak Signal-to-Noise Ratio in dB. Higher = better."""
    mse = np.mean((orig.astype(float) - processed.astype(float)) ** 2)
    if mse == 0:
        return 99.0
    return 20 * np.log10(255.0) - 10 * np.log10(mse)


def compute_ssim(orig, processed):
    """
    Structural Similarity Index (simplified single-channel).
    Range [0,1]. Higher = better.
    """
    a = (0.299 * orig[:,:,0] + 0.587 * orig[:,:,1] + 0.114 * orig[:,:,2]).astype(float)
    b = (0.299 * processed[:,:,0] + 0.587 * processed[:,:,1] + 0.114 * processed[:,:,2]).astype(float)
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.std(), b.std()
    sig_ab = np.mean((a - mu_a) * (b - mu_b))
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2) / \
           ((mu_a**2 + mu_b**2 + C1) * (sig_a**2 + sig_b**2 + C2))


# ─────────────────────────────────────────────
# 9. FFT VISUALISATION HELPER
# ─────────────────────────────────────────────

def fft_magnitude_map(img):
    """
    Compute log-scaled FFT magnitude spectrum (DC centred).
    Returns 2D float array ready for imshow.
    """
    gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    f = np.fft.fftshift(np.fft.fft2(gray))
    return np.log(1 + np.abs(f))


# ─────────────────────────────────────────────
# 10. VISUALISATION
# ─────────────────────────────────────────────

def visualise_results(orig, deg, spatial, output, alpha_map, dec_grid,
                      psnr_deg, psnr_sp, psnr_out, ssim_out,
                      save_path="nffs_output.png"):
    """
    Generate a comprehensive 3-row output figure:
        Row 1: original / degraded / spatial / hybrid (images)
        Row 2: FFT magnitude maps for all four
        Row 3: decision heatmap / alpha map / error map / PSNR bar chart
    """
    fig = plt.figure(figsize=(20, 14), facecolor='#0e1117')
    fig.suptitle(
        'Neural Frequency Fusion Studio — Full Pipeline Output',
        fontsize=16, color='white', fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.22)

    # ── Row 1: images ──
    row1_imgs   = [orig, deg, spatial, output]
    row1_titles = [
        'Original',
        f'Degraded  (σ=18, blur=1.5)',
        'Spatial Only  (unsharp mask)',
        'Hybrid Fusion Output'
    ]
    for i, (title, img) in enumerate(zip(row1_titles, row1_imgs)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_title(title, color='white', fontsize=9, fontweight='bold')
        ax.axis('off')
        p = compute_psnr(orig, img)
        ax.set_xlabel(f'PSNR: {p:.1f} dB', color='#aaaaaa', fontsize=8)
        ax.xaxis.set_label_position('bottom')

    # ── Row 2: FFT magnitude maps ──
    fft_cmap = LinearSegmentedColormap.from_list('fft', ['#000000', '#ff6000', '#ffdd00'])
    for i, (title, img) in enumerate(zip(
        ['FFT: Original', 'FFT: Degraded', 'FFT: Spatial', 'FFT: Hybrid'],
        row1_imgs
    )):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(fft_magnitude_map(img), cmap=fft_cmap)
        ax.set_title(title, color='white', fontsize=9)
        ax.axis('off')

    # ── Row 3 col 0: decision heatmap ──
    ax_h = fig.add_subplot(gs[2, 0])
    dec_cmap = LinearSegmentedColormap.from_list('dec', ['#378ADD', '#EF9F27', '#1D9E75'])
    ax_h.imshow(dec_grid, cmap=dec_cmap, vmin=0, vmax=2,
                interpolation='nearest', aspect='auto')
    rows_n, cols_n = dec_grid.shape
    for pr in range(rows_n):
        for pc in range(cols_n):
            lbl = ['S', 'H', 'F'][dec_grid[pr, pc]]
            ax_h.text(pc, pr, lbl, ha='center', va='center',
                      fontsize=7, color='white', fontweight='bold')
    ax_h.set_title('Decision Heatmap\n(S=spatial  H=hybrid  F=freq)',
                   color='white', fontsize=8)
    ax_h.axis('off')

    # ── Row 3 col 1: alpha / spatial-weight map ──
    ax_a = fig.add_subplot(gs[2, 1])
    ax_a.imshow(alpha_map, cmap='coolwarm', vmin=0, vmax=1)
    ax_a.set_title('Spatial Weight α per pixel\n(red=spatial  blue=freq)',
                   color='white', fontsize=8)
    ax_a.axis('off')

    # ── Row 3 col 2: residual error map ──
    ax_e = fig.add_subplot(gs[2, 2])
    err = np.abs(orig.astype(float) - output.astype(float)).mean(axis=2)
    ax_e.imshow(err, cmap='hot')
    ax_e.set_title('Residual Error Map\n(hybrid vs original)',
                   color='white', fontsize=8)
    ax_e.axis('off')

    # ── Row 3 col 3: PSNR bar chart ──
    ax_m = fig.add_subplot(gs[2, 3])
    ax_m.set_facecolor('#1a1d23')
    methods = ['Degraded', 'Spatial', 'Hybrid']
    psnrs   = [psnr_deg, psnr_sp, psnr_out]
    colors  = ['#E24B4A', '#378ADD', '#1D9E75']
    bars = ax_m.bar(methods, psnrs, color=colors, width=0.5, edgecolor='none')
    for bar, val in zip(bars, psnrs):
        ax_m.text(bar.get_x() + bar.get_width() / 2, val + 0.15,
                  f'{val:.1f}', ha='center', va='bottom',
                  color='white', fontsize=9, fontweight='bold')
    ax_m.set_ylim(min(psnrs) - 2, max(psnrs) + 3)
    ax_m.set_title(f'PSNR Comparison\nSSIM (hybrid) = {ssim_out:.3f}',
                   color='white', fontsize=8)
    ax_m.tick_params(colors='white', labelsize=8)
    for spine in ax_m.spines.values():
        spine.set_visible(False)
    ax_m.set_ylabel('dB', color='#aaaaaa', fontsize=8)
    ax_m.yaxis.label.set_color('#aaaaaa')

    for ax in fig.get_axes():
        ax.set_facecolor('#1a1d23')

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    print(f"[NFFS] Output saved → {save_path}")


# ─────────────────────────────────────────────
# 11. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(image_path=None, noise_sigma=18, blur_radius=1.5,
                 patch_size=32, cutoff=0.35, manual_alpha=None,
                 feedback_bias=0.0, output_path="nffs_output.png"):
    """
    End-to-end NFFS pipeline.

    Args:
        image_path    : str or None — path to input image (None → synthetic)
        noise_sigma   : float — Gaussian noise standard deviation
        blur_radius   : float — Gaussian blur kernel sigma
        patch_size    : int   — patch size for decision network (default 32)
        cutoff        : float — frequency low-pass cutoff (0–0.5)
        manual_alpha  : float or None — override network alpha [0,1]
        feedback_bias : float — reinforcement learning bias [-2, 2]
        output_path   : str   — where to save the output figure
    """
    print("[NFFS] ── Neural Frequency Fusion Studio ──")

    # Step 1: load / generate image
    if image_path:
        print(f"[NFFS] Loading image: {image_path}")
        orig = load_image(image_path)
    else:
        print("[NFFS] Generating synthetic test image (256×256)")
        orig = generate_synthetic_image(size=256)

    print(f"[NFFS] Image size: {orig.shape[1]}×{orig.shape[0]}")

    # Step 2: degradation
    print(f"[NFFS] Degrading: noise σ={noise_sigma}, blur radius={blur_radius}")
    deg = degrade_image(orig, noise_sigma=noise_sigma, blur_radius=blur_radius)

    # Step 3: dual-domain processing
    print("[NFFS] Spatial enhancement (unsharp mask / Laplacian sharpening)")
    spatial = spatial_enhance(deg)

    print(f"[NFFS] Frequency filtering (low-pass, cutoff={cutoff})")
    freq = frequency_filter(deg, cutoff=cutoff)

    # Step 4: patch decision + fusion
    print(f"[NFFS] Running decision network on {orig.shape[0]//patch_size}"
          f"×{orig.shape[1]//patch_size} patches (size={patch_size}px)")
    output, alpha_map, dec_grid, dec_info = hybrid_fusion(
        deg, spatial, freq,
        noise_sigma=noise_sigma,
        patch_size=patch_size,
        manual_alpha=manual_alpha,
        feedback_bias=feedback_bias
    )

    # Step 5: metrics
    psnr_deg = compute_psnr(orig, deg)
    psnr_sp  = compute_psnr(orig, spatial)
    psnr_out = compute_psnr(orig, output)
    ssim_out = compute_ssim(orig, output)

    print(f"\n[NFFS] ── Results ──")
    print(f"  PSNR degraded : {psnr_deg:.2f} dB")
    print(f"  PSNR spatial  : {psnr_sp:.2f} dB")
    print(f"  PSNR hybrid   : {psnr_out:.2f} dB  ← best")
    print(f"  SSIM hybrid   : {ssim_out:.4f}")

    # Step 6: count patch decisions
    modes = {0: "spatial", 1: "hybrid", 2: "freq"}
    for code, name in modes.items():
        cnt = int((dec_grid == code).sum())
        print(f"  Patches → {name:8s}: {cnt}")

    # Step 7: visualise
    print("\n[NFFS] Generating output figure …")
    visualise_results(
        orig, deg, spatial, output, alpha_map, dec_grid,
        psnr_deg, psnr_sp, psnr_out, ssim_out,
        save_path=output_path
    )

    return {
        "original": orig, "degraded": deg, "spatial": spatial,
        "output": output, "alpha_map": alpha_map, "dec_grid": dec_grid,
        "dec_info": dec_info,
        "metrics": {"psnr_deg": psnr_deg, "psnr_spatial": psnr_sp,
                    "psnr_hybrid": psnr_out, "ssim": ssim_out}
    }


# ─────────────────────────────────────────────
# 12. CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Frequency Fusion Studio")
    parser.add_argument("--image",   type=str,   default=None,
                        help="Path to input image (omit to use synthetic image)")
    parser.add_argument("--noise",   type=float, default=18.0,
                        help="Gaussian noise sigma (default: 18)")
    parser.add_argument("--blur",    type=float, default=1.5,
                        help="Gaussian blur radius (default: 1.5)")
    parser.add_argument("--patch",   type=int,   default=32,
                        help="Patch size in pixels (default: 32)")
    parser.add_argument("--cutoff",  type=float, default=0.35,
                        help="Frequency low-pass cutoff 0.0–0.5 (default: 0.35)")
    parser.add_argument("--alpha",   type=float, default=None,
                        help="Manual spatial alpha override 0.0–1.0 (default: auto)")
    parser.add_argument("--bias",    type=float, default=0.0,
                        help="Feedback bias for decision network (default: 0.0)")
    parser.add_argument("--output",  type=str,   default="nffs_output.png",
                        help="Output figure path (default: nffs_output.png)")
    args = parser.parse_args()

    run_pipeline(
        image_path    = args.image,
        noise_sigma   = args.noise,
        blur_radius   = args.blur,
        patch_size    = args.patch,
        cutoff        = args.cutoff,
        manual_alpha  = args.alpha,
        feedback_bias = args.bias,
        output_path   = args.output,
    )
