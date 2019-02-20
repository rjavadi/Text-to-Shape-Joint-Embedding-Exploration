import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
from PIL import Image

img_h = img_w = 40


def open_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data


def resize_images(basedir, model_ids):
    for id in model_ids:
        im_path = os.path.join(basedir, id, id + '.png')
        im_save_path = os.path.join(basedir, "resize_models")
        fname = im_save_path + """\\""" + id + '.png'
        print("image: ", id)
        if not os.path.isfile(fname):
            print(fname, "does not exist")
            img = Image.open(im_path)
            print(np.shape(img))
            img = img.resize((img_w, img_h), Image.ANTIALIAS)
            img.save(fname)


def get_images(img_dir):
    img_data = []
    images_list = os.listdir(img_dir)
    for name in images_list:
        if name.__contains__('.png'):
            path = os.path.join(img_dir, name)
            img_data.append(np.array(Image.open(path)))
    return img_data


def write_sprite_image(filename, images):
    """
        Create a sprite image consisting of sample images
        :param filename: name of the file to save on disk
        :param images: tensor of flattened images
    """

    # Calculate number of plot
    n_plots = int(np.ceil(np.sqrt(len(images))))

    # Make the background of sprite image
    sprite_image = np.ones((img_h * n_plots, img_w * n_plots, 4))

    for i in range(n_plots):
        for j in range(n_plots):
            img_idx = i * n_plots + j
            if img_idx < len(images):
                img = images[img_idx]
                for k in range(4):
                    sprite_image[i * img_h:(i + 1) * img_h,
                    j * img_w:(j + 1) * img_w, k] = img

    plt.imsave(filename, sprite_image, cmap='gray')
    print('Sprite image saved in {}'.format(filename))


def images_to_sprite(data, path_to_sprite):
    """Creates the sprite image along with any necessary padding
    Args:
    data: NxHxW[x3] tensor containing the images.
    Returns:
    data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
    # Inverting the colors seems to look better for MNIST
    # data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    plt.imsave(path_to_sprite, data)
    return data


def write_metadata(filename, labels):
    """
            Create a metadata file image consisting of sample indices and labels
            :param filename: name of the file to save on disk
            :param shape: tensor of labels
    """
    with open(filename, 'w') as f:
        # f.write("Index\tLabel\n")
        for index, label in enumerate(labels):
            f.write("{}\n".format(label))

    print('Metadata file saved in {}'.format(filename))




def visualize_embeddings(embeddings, basedir, metadata):
    with tf.Session() as sess:
        embedding_var = tf.Variable(embeddings, name='embeddings')
        sess.run(embedding_var.initializer)
        init = tf.global_variables_initializer()
        init.run()

        config = projector.ProjectorConfig()
        config.model_checkpoint_path = os.path.join(basedir, "tsne", 'my-model.ckpt')
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = os.path.join(os.path.join(basedir, "tsne", metadata))
        embedding.sprite.image_path = os.path.join(basedir, "tsne", "sprite.jpg")
        embedding.sprite.single_image_dim.extend([img_w, img_h])
