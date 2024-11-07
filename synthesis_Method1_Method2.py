import cv2
import numpy as np
import random
import argparse


class TextureSynthesizer:
    def __init__(self, input_texture_path, patch_size, overlap_size, tolerance=1.1):
        self.input_texture_path = input_texture_path
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.input_texture = cv2.imread(input_texture_path)
        if self.input_texture is None:
            raise ValueError("Texture image not found.")
        self.scale_factor = 5
        self.tolerance = tolerance

    def synthesize_texture(self, method='Method1'):
        if method == 'Method1':
            return self.method_1()
        elif method == 'Method2':
            return self.method_2()
        else:
            raise ValueError("Invalid method selected.")

    def method_1(self):
        input_height, input_width, _ = self.input_texture.shape
        output_height = input_height * self.scale_factor
        output_width = input_width * self.scale_factor
        output_texture = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        for y in range(0, output_height, self.patch_size):
            for x in range(0, output_width, self.patch_size):
                patch = self._extract_random_patch()
                # Calculate the end coordinates for the patch
                end_y = y + self.patch_size
                end_x = x + self.patch_size

                # Adjust the end coordinates if they exceed the output texture's dimensions
                end_y = min(end_y, output_height)
                end_x = min(end_x, output_width)

                # Adjust the patch size if necessary (for edges)
                adjusted_patch = patch[:end_y - y, :end_x - x]

                # Place the adjusted patch into the output texture
                output_texture[y:end_y, x:end_x] = adjusted_patch

        return output_texture

    def _extract_random_patch(self):

        max_y = self.input_texture.shape[0] - self.patch_size
        max_x = self.input_texture.shape[1] - self.patch_size
        y = random.randint(0, max_y)
        x = random.randint(0, max_x)
        return self.input_texture[y:y + self.patch_size, x:x + self.patch_size]



    def method_2(self):

        input_height, input_width, _ = self.input_texture.shape
        output_height = input_height * self.scale_factor
        output_width = input_width * self.scale_factor
        output_texture = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        for y in range(0, output_height, self.patch_size - self.overlap_size):
            for x in range(0, output_width, self.patch_size - self.overlap_size):
                if y == 0 and x == 0:
                    # For the first patch, just select a random one
                    patch = self._extract_random_patch()
                else:
                    patch = self._find_best_patch(output_texture, y, x)


                space_y = output_texture.shape[0] - y
                space_x = output_texture.shape[1] - x


                adjusted_patch_height = min(self.patch_size, space_y)
                adjusted_patch_width = min(self.patch_size, space_x)


                adjusted_patch = patch[:adjusted_patch_height, :adjusted_patch_width]


                output_texture[y:y + adjusted_patch_height, x:x + adjusted_patch_width] = adjusted_patch

        return output_texture

    def _calculate_ssd(self, patch, output, pos_y, pos_x):
        """Calculates the SSD for a given position and patch."""
        ssd = 0
        if pos_y > 0:
            existing_overlap = output[pos_y:pos_y + self.overlap_size, pos_x:pos_x + self.patch_size]
            patch_overlap = patch[:self.overlap_size, :]

            min_width = min(existing_overlap.shape[1], patch_overlap.shape[1])
            ssd += np.sum((existing_overlap[:, :min_width] - patch_overlap[:, :min_width]) ** 2)

        if pos_x > 0:
            existing_overlap = output[pos_y:pos_y + self.patch_size, pos_x:pos_x + self.overlap_size]
            patch_overlap = patch[:, :self.overlap_size]

            min_height = min(existing_overlap.shape[0], patch_overlap.shape[0])
            ssd += np.sum((existing_overlap[:min_height, :] - patch_overlap[:min_height, :]) ** 2)

        return ssd

    def _find_best_patch(self, output, pos_y, pos_x):

        h, w, _ = self.input_texture.shape
        candidates = []
        for i in range(h - self.patch_size + 1):
            for j in range(w - self.patch_size + 1):
                patch = self.input_texture[i:i + self.patch_size, j:j + self.patch_size]
                ssd = self._calculate_ssd(patch, output, pos_y, pos_x)
                candidates.append((ssd, patch))

        candidates.sort(key=lambda x: x[0])
        min_error = candidates[0][0]

        best_patches = [patch for ssd, patch in candidates if ssd <= min_error * self.tolerance]
        return random.choice(best_patches)




def main():
    parser = argparse.ArgumentParser(description="Texture Synthesis with different methods.")
    parser.add_argument("--input_texture_path", type=str, required=True, help="Path to the input texture image")
    parser.add_argument("--patch_size", type=int, required=True, help="Size of the texture patches")
    parser.add_argument("--overlap_size", type=int, required=True, help="Size of the overlapping region")
    parser.add_argument("--method", type=str, choices=['Method1', 'Method2'], required=True,
                        help="Synthesis method to use")
    args = parser.parse_args()

    synthesizer = TextureSynthesizer(args.input_texture_path, args.patch_size, args.overlap_size)
    output_texture = synthesizer.synthesize_texture(method=args.method)
    output_file = f"output_texture_{args.method}.jpeg"
    cv2.imwrite(output_file, output_texture)

    print(f"Output texture saved as {output_file}")


if __name__ == "__main__":
    main()
