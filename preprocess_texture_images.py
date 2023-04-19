import os
from skimage import io, util, transform

# First crops to 1:1 ratio, then resizes to img_size.
def preprocess(target_folder, final_folder):
    img_size = (256,256)
    total_count = 0
    counter = 0
    for root, dirs, files in os.walk(target_folder):
        for file in files:
            if file.endswith('.jpg'):
                total_count += 1
    for root, dirs, files in os.walk(target_folder):
        for file in files:
            if file.endswith('.jpg'):
                counter += 1
                file_path = os.path.join(root, file)
                image = io.imread(file_path)
                cropped_image = util.crop(image, ((0, 0), (0, image.shape[1] - image.shape[0]), (0, 0)))
                resized_image = (transform.resize(image=cropped_image, output_shape=img_size, anti_aliasing=True) * 255).astype('uint8')
                resized_path = os.path.join(final_folder, file)
                io.imsave(resized_path, resized_image)
                print("resized:", file, "   ", counter, "/", total_count)

def main():
    img_folder = "/Users/gabrielmahler/Documents/BrownCS/1470/dtd"
    final_path = "/Users/gabrielmahler/Documents/BrownCS/1470/Texture-Sythesis-with-CNN/preprocessed_images"
    preprocess(img_folder, final_path)

if __name__== "__main__":
    main()