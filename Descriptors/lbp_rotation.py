#!/usr/bin/env python
# ultra_simple_lbp_rotation.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from lbp import FeatureLBP 
from matplotlib.lines import Line2D

def preprocess(img):
    img = np.asarray(img, dtype=np.float32)
    img = np.pad(img, pad_width=5, mode="constant", constant_values=1.0)
    img = transform.rescale(img, 2.0, order=1, preserve_range=True, anti_aliasing=False)
    return img.astype(np.float32)

def rotate_img(img, angle_deg):
    return transform.rotate(img, angle=angle_deg, resize=False, mode="constant",
                            cval=1.0, preserve_range=True).astype(np.float32)

def to_uint8(img):
    if img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

def chi2_distance(a, b, eps=1e-9): #to see similarity
    return 0.5 * np.sum((a - b) ** 2 / (a + b + eps))

"""
Now we'll do the exercise 4 c), we need to plot it in 2D.

"""
def pca_2d(X):
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    vals, vecs = np.linalg.eigh(C)
    W = vecs[:, np.argsort(vals)[::-1][:2]]  # top-2 eigenvectors
    return Xc @ W


def scatter_2d(Z, digits, angles, title):
    markers = {"6": "o", "9": "s"}
    colors  = {"0": "C0", "90": "C1", "180": "C2", "270": "C3"}
    plt.figure(figsize=(5.6, 4.4))
    for (x, y), d, a in zip(Z, digits, angles):
        plt.scatter(x, y, marker=markers[d], c=colors[str(a)], s=70, edgecolor="k", linewidths=0.6, label=f"{d}@{a}")

    leg_items = {}
    for d in ("6","9"):
        for a in ("0","90","180","270"):
            key = f"{d}@{a}"
            if key not in leg_items:
                leg_items[key] = Line2D([0],[0], marker=markers[d], color='w',
                                        markerfacecolor=colors[a], markeredgecolor='k',
                                        linestyle='', markersize=8, label=key)
    plt.legend(leg_items.values(), leg_items.keys(), bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.title(title); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.show()




if __name__ == "__main__":
    # Load data
    here = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(here, "../datasets/example_fonts/digitsIms.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    digits = data.get("digitsIms", next(v for v in data.values())) if isinstance(data, dict) else data

    # Prepare images for digit 6 and 9
    img6 = preprocess(digits[6])
    img9 = preprocess(digits[9])
    angles = [0, 90, 180, 270]

    stack6 = [rotate_img(img6, a) for a in angles]
    stack9 = [rotate_img(img9, a) for a in angles]

    # Create two LBP extractors: non-invariant and invariant
    lbp_default = FeatureLBP(radius=1, method="default", lbp_type="simple")  # NOT rotation invariant
    lbp_uniform = FeatureLBP(radius=1, method="uniform", lbp_type="simple")  # rotation invariant

    # Compute descriptors (histograms) for each stack
    H6_def, H9_def, H6_uni, H9_uni = [], [], [], []

    for im in stack6:
        H6_def.append(lbp_default.extract_image_features(to_uint8(im)))
        H6_uni.append(lbp_uniform.extract_image_features(to_uint8(im)))

    for im in stack9:
        H9_def.append(lbp_default.extract_image_features(to_uint8(im)))
        H9_uni.append(lbp_uniform.extract_image_features(to_uint8(im)))
    """
    # Exercise 4 b)
    # Print comparisons (χ² = difference score; 0 = identical)
    print("\n[Non-invariant] method=default")
    print("6: ",
          "0° vs 90° =", f"{chi2_distance(H6_def[0], H6_def[1]):.4f}",
          "| 0° vs 180° =", f"{chi2_distance(H6_def[0], H6_def[2]):.4f}",
          "| 0° vs 270° =", f"{chi2_distance(H6_def[0], H6_def[3]):.4f}")
    print("9: ",
          "0° vs 90° =", f"{chi2_distance(H9_def[0], H9_def[1]):.4f}",
          "| 0° vs 180° =", f"{chi2_distance(H9_def[0], H9_def[2]):.4f}",
          "| 0° vs 270° =", f"{chi2_distance(H9_def[0], H9_def[3]):.4f}")
    print("6 vs 9 (same angle):",
          ", ".join([f"{ang}°:{chi2_distance(H6_def[i], H9_def[i]):.4f}" for i, ang in enumerate(angles)]))
    print("6@0° vs 9@180°:", f"{chi2_distance(H6_def[0], H9_def[2]):.4f}")

    print("\n[Rotation-invariant] method=uniform")
    print("6: ",
          "0° vs 90° =", f"{chi2_distance(H6_uni[0], H6_uni[1]):.4f}",
          "| 0° vs 180° =", f"{chi2_distance(H6_uni[0], H6_uni[2]):.4f}",
          "| 0° vs 270° =", f"{chi2_distance(H6_uni[0], H6_uni[3]):.4f}")
    print("9: ",
          "0° vs 90° =", f"{chi2_distance(H9_uni[0], H9_uni[1]):.4f}",
          "| 0° vs 180° =", f"{chi2_distance(H9_uni[0], H9_uni[2]):.4f}",
          "| 0° vs 270° =", f"{chi2_distance(H9_uni[0], H9_uni[3]):.4f}")
    print("6 vs 9 (same angle):",
          ", ".join([f"{ang}°:{chi2_distance(H6_uni[i], H9_uni[i]):.4f}" for i, ang in enumerate(angles)]))
    print("6@0° vs 9@180°:", f"{chi2_distance(H6_uni[0], H9_uni[2]):.4f}")

    # Plot LBP maps (image, default, uniform) for 6 and 9
    def plot_maps(label, stack):
        fig, axes = plt.subplots(4, 3, figsize=(9, 8))
        fig.suptitle(f"Digit {label}: image + LBP maps")
        for i in range(4):
            axes[i, 0].imshow(stack[i], cmap="gray")
            axes[i, 0].set_title(f"{angles[i]}°"); axes[i, 0].axis("off")

            lbp_def_img = lbp_default.extract_pixel_features(to_uint8(stack[i]))[:, :, 0]
            axes[i, 1].imshow(lbp_def_img, cmap="gray")
            axes[i, 1].set_title("LBP default"); axes[i, 1].axis("off")

            lbp_uni_img = lbp_uniform.extract_pixel_features(to_uint8(stack[i]))[:, :, 0]
            axes[i, 2].imshow(lbp_uni_img, cmap="gray")
            axes[i, 2].set_title("LBP uniform"); axes[i, 2].axis("off")
        plt.tight_layout(); plt.show()

    plot_maps("6", stack6)
    plot_maps("9", stack9)

    # Plot histograms across rotations (only 6)
    def plot_hist_row(title, hist_list):
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        fig.suptitle(title)
        for i in range(4):
            axs[i].bar(np.arange(len(hist_list[i])), hist_list[i])
            axs[i].set_title(f"{angles[i]}°")
            axs[i].set_xticks([]); axs[i].set_xlim(0, len(hist_list[i]))
        plt.tight_layout(); plt.show()

    plot_hist_row("Digit 9 histograms — Non-invariant (default)", H9_def)
    plot_hist_row("Digit 9 histograms — Rotation-invariant (uniform)", H9_uni)
    """
    # Uniform LBP + BLOCK-WISE (keeps layout)
    lbp_uniform_block = FeatureLBP(radius=1, method="uniform", lbp_type="block_lbp")  # default grid (4,4)
    X_block, y_digit, y_angle = [], [], []
    for d, stack in (("6", stack6), ("9", stack9)):
        for a, im in zip(angles, stack):
            X_block.append(lbp_uniform_block.extract_image_features(to_uint8(im)))
            y_digit.append(d); y_angle.append(a)
    X_block = np.vstack(X_block)
    Z_block = pca_2d(X_block)
    scatter_2d(Z_block, y_digit, y_angle, "Uniform LBP + Block-wise histogram (4×4)")

    # 2) Uniform LBP + GLOBAL (1×1, order-free)
    lbp_uniform_global = FeatureLBP(radius=1, method="uniform", lbp_type="simple")
    X_glob, y_digit_g, y_angle_g = [], [], []
    for d, stack in (("6", stack6), ("9", stack9)):
        for a, im in zip(angles, stack):
            X_glob.append(lbp_uniform_global.extract_image_features(to_uint8(im)))
            y_digit_g.append(d); y_angle_g.append(a)
    X_glob = np.vstack(X_glob)
    Z_glob = pca_2d(X_glob)
    scatter_2d(Z_glob, y_digit_g, y_angle_g, "Uniform LBP + Global histogram (1×1)")
