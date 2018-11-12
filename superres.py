from numpy import array, load, concatenate
from os import makedirs, remove
from scipy.misc import imsave, imresize, imread
from skimage import io
from glob import glob
from time import strftime
from model import generator as generator
import tensorflow as tf
import numpy as np

model = 'default'
dataset_size = 1
batch_size = 1
out_path = 'downloads/'
odims = [64, 64]
ndims_l = [128, 128]

def get_dataset():
    print('Importing test set ...')
    files = glob('uploads/*.*')
    dataset = np.array([imread(file) for file in files])
    print('Done.')
    dataset_size  = dataset.shape[0]
    print('Converting from {} to {}'.format(odims, ndims_l))
    return dataset

def test(dataset):
    sml_x   = tf.placeholder(tf.float32, [None,  None,  None, 3])
    ndims   = tf.placeholder(tf.int32, [2])
    gener_x = generator(sml_x, ndims, is_training=False, reuse=False)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        try:
            saver.restore(sess, '/'.join(['models', model, model]))
        except Exception as e:
            print('Model coult not be restored. Exiting.\nError: ' + e)
            exit()

        print('Saving test results ...')
        start = 0
        batch_big = dataset / 255.0
        batch_sml = array([imresize(img, size=(64, 64, 3)) for img in batch_big])
        superres_imgs = sess.run(gener_x, feed_dict={sml_x: batch_sml, ndims: ndims_l})
        images = concatenate(
            (
                array([imresize(img, size=superres_imgs.shape[1:], interp='lanczos')/255.0 for img in batch_sml]),
                superres_imgs
            ), 2)
        for idx, image in enumerate(images):
            imsave('static/output/abc.png', image)
        start += batch_size
        print('%d/%d saved successfully.' % (min(start, dataset_size), dataset_size))

def clean_files():
    files = glob('uploads/*.*') + glob('static/output/*.*')
    for file in files:
        remove(file)

if __name__ == '__main__':
    dataset = get_dataset()
    test(dataset)
