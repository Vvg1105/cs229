from __future__ import division, print_function
import argparse
import re
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    height, width, channels = image.shape
    
    # Each row is a pixel, and each column is a channel
    pixels = image.reshape(height * width, channels).astype(np.float64)

    random_indices = np.random.choice(height * width, size=num_clusters, replace=False)

    centroid_pixels = pixels[random_indices]

    return centroid_pixels
    # *** END YOUR CODE ***

def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    height, width, channels = image.shape
    
    # Convert to float to avoid integer rounding issues
    pixels = image.reshape(height * width, channels).astype(np.float64)

    num_clusters = centroids.shape[0]

    new_centroids = centroids.copy()

    for i in range(max_iter):

        distances = np.zeros((pixels.shape[0], num_clusters))

        for cluster in range(num_clusters):

            distance = np.linalg.norm(pixels - new_centroids[cluster], axis=1)

            distances[:, cluster] = distance
        
        pixel_assign = np.argmin(distances, axis=1)

        old_centroids = np.copy(new_centroids)

        for cluster in range(num_clusters):

            pixels_in_cluster = pixels[pixel_assign == cluster]

            if len(pixels_in_cluster) > 0:
                new_centroids[cluster] = pixels_in_cluster.mean(axis=0)
        
        if np.allclose(old_centroids, new_centroids):
            break

        if (i + 1) % print_every == 0:
            print(f'Iteration {i + 1}/{max_iter} completed')

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    height, width, channels = image.shape

    for h in range(height):
        for w in range(width):
            pixel = image[h, w]

            distances = np.linalg.norm(pixel - centroids, axis=1)

            closest_centroid_idx = np.argmin(distances)

            image[h, w] = centroids[closest_centroid_idx]

    return image

def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
