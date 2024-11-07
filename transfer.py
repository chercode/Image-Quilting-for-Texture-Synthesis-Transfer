import numpy as np
import cv2
import argparse
import os
import heapq

def calculate_overlap_energy(patch, block_size, overlap, res, y, x):
    energy = 0
    if x > 0:
        left = patch[:, :overlap] - res[y:y + block_size, x:x + overlap]
        energy += np.sum(left**2)
    if y > 0:
        up = patch[:overlap, :] - res[y:y + overlap, x:x + block_size]
        energy += np.sum(up**2)
    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - res[y:y + overlap, x:x + overlap]
        energy -= np.sum(corner**2)
    return energy

def find_min_cut_path(energies):
    pq = [(energy, [i]) for i, energy in enumerate(energies[0])]
    heapq.heapify(pq)
    height, width = energies.shape
    seen = set()
    while pq:
        energy, path = heapq.heappop(pq)
        cur_depth = len(path)
        cur_index = path[-1]
        if cur_depth == height:
            return path
        for delta in -1, 0, 1:
            next_index = cur_index + delta
            if 0 <= next_index < width and (cur_depth, next_index) not in seen:
                cum_energy = energy + energies[cur_depth, next_index]
                heapq.heappush(pq, (cum_energy, path + [next_index]))
                seen.add((cur_depth, next_index))
    return []

def apply_min_cut_patch(patch, block_size, overlap, res, y, x):
    patch = patch.copy()
    dy, dx, _ = patch.shape
    min_cut = np.zeros_like(patch, dtype=bool)
    if x > 0:
        left = patch[:, :overlap] - res[y:y + dy, x:x + overlap]
        left_l2 = np.sum(left ** 2, axis=2)
        for i, j in enumerate(find_min_cut_path(left_l2)):
            min_cut[i, :j] = True
    if y > 0:
        up = patch[:overlap, :] - res[y:y + overlap, x:x + dx]
        up_l2 = np.sum(up ** 2, axis=2)
        for j, i in enumerate(find_min_cut_path(up_l2.T)):
            min_cut[:i, j] = True
    patch[min_cut] = res[y:y + dy, x:x + dx][min_cut]
    return patch

def calculate_correspondence_error(patch, target_patch, alpha):

    return alpha * np.sum((patch - target_patch) ** 2)

def select_best_patch(texture, target, block_size, overlap, res, y, x, alpha):

    h, w, _ = texture.shape
    best_error = np.inf
    best_patch = None
    for i in range(h - block_size):
        for j in range(w - block_size):
            patch = texture[i:i+block_size, j:j+block_size]
            target_patch = target[y:y+block_size, x:x+block_size]
            overlap_error = calculate_overlap_energy(patch, block_size, overlap, res, y, x)
            correspondence_error = calculate_correspondence_error(patch, target_patch, 1 - alpha)
            total_error = alpha * overlap_error + correspondence_error
            if total_error < best_error:
                best_error = total_error
                best_patch = patch
    return best_patch

def quilt_texture_with_transfer(texture_path, target_path, output_path, block_size, alpha):
    texture = cv2.imread(texture_path).astype(np.float32)
    target = cv2.imread(target_path).astype(np.float32)
    assert texture is not None and target is not None, "Image paths are invalid."
    overlap = block_size // 6
    h, w, _ = target.shape
    res = np.zeros_like(target)

    for y in range(0, h - block_size + 1, block_size - overlap):
        for x in range(0, w - block_size + 1, block_size - overlap):
            patch = select_best_patch(texture, target, block_size, overlap, res, y, x, alpha)
            patch = apply_min_cut_patch(patch, block_size, overlap, res, y, x)
            res[y:y+block_size, x:x+block_size] = patch
    if not output_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        output_path += '.jpg'  # Defaulting to JPG if no valid extension is provided

    cv2.imwrite(output_path, res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Texture Transfer using Patch-Based Synthesis')
    parser.add_argument('--texture_path', required=True, help='Path to the texture image.')
    parser.add_argument('--target_path', required=True, help='Path to the target image.')
    parser.add_argument('--output_path', required=True, help='Output path for the transferred image.')
    parser.add_argument('--block_size', type=int, default=50, help='Block size for texture patches.')
    parser.add_argument('--alpha', type=float, default=0.8, help='Alpha value to balance overlap and correspondence error.')
    args = parser.parse_args()

    quilt_texture_with_transfer(args.texture_path, args.target_path, args.output_path, args.block_size, args.alpha)
