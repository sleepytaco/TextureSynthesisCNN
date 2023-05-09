from PIL import Image
from torchvision import transforms
from skimage import io, transform, util
import numpy as np
import os

"""
Contains utility functions to work with images in tensor and jpg/png forms
"""


def load_image_tensor(path):
    """
    Returns Image as a Pytorch Tensor of shape ((img_size),3).
    Values between 0 and 1.
    """
    img_size = (256, 256)
    image = io.imread(path)
    cropped_image = util.crop(image, ((0, 0), (0, image.shape[1] - image.shape[0]), (0, 0)))
    resized_image = (transform.resize(image=cropped_image, output_shape=img_size, anti_aliasing=True))
    to_tensor = transforms.Compose([transforms.ToTensor()])
    tensor = to_tensor(resized_image)
    # tensor = tensor.permute(1,2,0)  # the model expects w, h, 3!
    return tensor.float()


def save_image_tensor(tensor, output_dir="./", image_name="output.png"):
    """
    Saves a 3D tensor as an image.
    """
    output_image = tensor.numpy().transpose(1, 2, 0)
    output_image = np.clip(output_image, 0, 1) * 255
    output_image = output_image.astype(np.uint8)
    output_image = Image.fromarray(output_image)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_image.save(output_dir + image_name)

    return output_image


def display_image_tensor(tensor):
    """
    Displays the passed in 3D image tensor
    """
    output_image = tensor.numpy().transpose(1, 2, 0)
    output_image = np.clip(output_image, 0, 1) * 255
    output_image = output_image.astype(np.uint8)
    output_image = Image.fromarray(output_image)
    output_image.show()


def get_grayscale(tensor):
    """
    Converts a 3D image tensor to greyscale
    """
    greyscale_transform = transforms.Grayscale()
    return greyscale_transform(tensor)
