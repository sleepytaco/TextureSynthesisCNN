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


def save_image_tensor(tensor):
    """
    Saves a tensor as an image.
    """
    generated_image = tensor.numpy().transpose(1, 2, 0)
    generated_image = np.clip(generated_image, 0, 1) * 255
    generated_image = generated_image.astype(np.uint8)
    generated_image = Image.fromarray(generated_image)
    generated_image.save('output.png')
    return
    numpy_image = (tensor.numpy()*255).astype(np.uint8)
    cwd = os.getcwd()
    filepath = os.path.join(cwd, "output_image.png")
    io.imsave(filepath, numpy_image)


def reparametrize_image(image):
    min_val = image.min()
    val_range = image.max() - min_val
    return lambda x: ((1.0 / (1.0 + torch.exp(-x))) * val_range) + min_val


def calculate_gram_matrices(feature_maps):
    gram_matrices = []
    for fm in feature_maps:
        b = 1
        n, x, y = fm.size()
        act = fm.view(b * n, x * y)
        gram_mat = torch.mm(act, act.t())
        gram_matrices.append(gram_mat.div(b*n*x*y))
    return gram_matrices
