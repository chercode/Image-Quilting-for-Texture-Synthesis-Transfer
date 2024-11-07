import numpy as np
import cv2
import argparse
import os
import heapq

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", required=True, type=str)
parser.add_argument("--block_size", type=int, default=50)
parser.add_argument("--mode", type=str, default='Method_3')
args = parser.parse_args()

def select_random_patch(texture, block_size):

   height, width, _ = texture.shape
   i = np.random.randint(height - block_size)
   j = np.random.randint(width - block_size)

   return texture[i:i + block_size, j:j + block_size]

def calculate_overlap_energy(patch, block_size, overlap, res, y, x):


    energy = 0
    if x>0:
        left = patch[:, :overlap] - res[y:y + block_size, x:x + overlap]
        energy += np.sum(left**2)

    if y>0:
        up = patch[:overlap, :] - res[y:y + overlap, x:x + block_size]
        energy += np.sum(up**2)

    if x>0 and y>0:
        corner = patch[:overlap, :overlap] - res[y:y + overlap, x:x + overlap]
        energy -= np.sum(corner**2)


    return energy

def select_best_patch(texture, block_size, overlap, res, y, x):


   height, width, _ = texture.shape

   energies = np.zeros((height - block_size, width - block_size))

   for i in range(height - block_size):
       for j in range(width - block_size):
           patch = texture[i:i + block_size, j:j + block_size]
           energy = calculate_overlap_energy(patch, block_size, overlap, res, y, x)
           energies[i, j] = energy
   i, j = np.unravel_index(np.argmin(energies), energies.shape)


   return texture[i:i + block_size, j:j + block_size]

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

def apply_min_cut_patch(patch, block_size, overlap, res, y, x):


    patch = patch.copy()
    dy, dx, _ = patch.shape
    min_cut = np.zeros_like(patch, dtype=bool)

    if x>0:
        left = patch[:, :overlap] - res[y:y + dy, x:x + overlap]
        left_l2 = np.sum(left ** 2, axis=2)
        for i, j in enumerate(find_min_cut_path(left_l2)):
            min_cut[i, :j] = True
    if y>0:
        up = patch[:overlap, :] - res[y:y + overlap, x:x + dx]
        up_l2 = np.sum(up ** 2, axis=2)
        for j, i in enumerate(find_min_cut_path(up_l2.T)):
            min_cut[:i, j] = True
    patch[min_cut] = res[y:y + dy, x:x + dx][min_cut]


    return patch

def quilt_texture(image_path, block_size, mode, sequence=False):
    texture = cv2.imread(image_path).astype(np.float32) / 255.0
    texture_height, texture_width, _ = texture.shape
    overlap = block_size // 6

    new_height = texture_height * 5
    new_width = texture_width * 5

    num_blocks_high = (new_height + (overlap - block_size)) // (block_size - overlap)
    num_blocks_wide = (new_width + (overlap - block_size)) // (block_size - overlap)

    height = (num_blocks_high * block_size) - (num_blocks_high - 1) * overlap
    width = (num_blocks_wide * block_size) - (num_blocks_wide - 1) * overlap

    res = np.zeros((height, width, texture.shape[2]))
    for i in range(num_blocks_high):
        for j in range(num_blocks_wide):
            y = i * (block_size - overlap)
            x = j * (block_size - overlap)

            patch = select_best_patch(texture, block_size, overlap, res, y, x)
            patch = apply_min_cut_patch(patch, block_size, overlap, res, y, x)
            res[y:y + block_size, x:x + block_size] = patch
    return (res * 255).astype(np.uint8)

if __name__ == "__main__":
    image_path = args.image_path
    block_size = args.block_size
    mode = args.mode
    result_image = quilt_texture(image_path, block_size, mode)

    output_folder = "result_sythesis"

    os.makedirs(output_folder, exist_ok=True)

    output_image_path = os.path.join(output_folder, "text_150_method3.jpeg")

    cv2.imwrite(output_image_path, result_image)
