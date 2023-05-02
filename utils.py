import torch
from PIL import Image
from torchvision import transforms
from skimage import io, transform, util
import numpy as np
import os


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
    Saves a tensor as an image.
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
    Displays the passed in image tensor
    """
    output_image = tensor.numpy().transpose(1, 2, 0)
    output_image = np.clip(output_image, 0, 1) * 255
    output_image = output_image.astype(np.uint8)
    output_image = Image.fromarray(output_image)
    output_image.show()


def calculate_gram_matrices(feature_maps):
    """
    Takes in an array of feature maps and converts each feature map into its gram matrix
    """
    gram_matrices = []
    for fm in feature_maps:
        n, x, y = fm.size()
        act = fm.view(n, x * y)
        gram_mat = torch.mm(act, act.t())
        gram_matrices.append(gram_mat.div(4. * n * x * y))
    return gram_matrices


def get_i_tilde(i, i_hat):
    i_hat = get_grayscale(i_hat)
    i = get_grayscale(i)
    fourier_i = torch.fft.fft2(i)
    fourier_i_hat = torch.fft.fft2(i_hat)
    conj_fourier_i = torch.conj(fourier_i)
    epsilon = 10e-12
    result = torch.fft.ifft2(
        (fourier_i_hat * conj_fourier_i) / (torch.abs(fourier_i_hat * conj_fourier_i) + epsilon) * fourier_i)
    return result


def get_grayscale(tensor):
    greyscale_transform = transforms.Grayscale()
    return greyscale_transform(tensor)
