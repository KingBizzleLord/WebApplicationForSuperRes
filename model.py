import tensorflow as tf

from numpy.random import permutation

def conv2d(x, output_dim, kernel=3, stride=2, padding='SAME'):
    return tf.layers.conv2d(x, output_dim, [kernel, kernel], strides=(stride, stride), padding=padding)

def dense(x,output_size, activation=tf.nn.relu):
    return tf.layers.dense(x , output_size , activation)

def lrelu(x, threshold=0.01):
    return tf.maximum(x, threshold*x)


def generator(x, new_dims, is_training=True, reuse=False):
    with tf.variable_scope('generator') as scope:
        if reuse:            
            scope.reuse_variables()
        conv1 = conv2d(x, output_dim=32, stride=1)
        conv1 = lrelu(conv1)
        #64 x 64 x 32
        
        conv2 = conv2d(conv1, output_dim=128, stride=1)
        conv2 = lrelu(conv2)
        #64 x 64 x 128

        conv3 = conv2d(conv2, output_dim=128, stride=1)
        conv3 = lrelu(conv3)
        #64 x 64 x 128

        resize = tf.image.resize_images(conv3, size=new_dims)

        conv4 = conv2d(resize, output_dim=128, stride=1)
        conv4 = lrelu(conv4)
        #128 x 128 x 128

        conv5 = conv2d(conv4, output_dim=64, stride=1)
        conv5 = lrelu(conv5)
        #128 x 128 x 64

        conv6 = conv2d(conv5, output_dim=3, stride=1)
        conv6 = tf.nn.sigmoid(conv6)
        #128 x 128 x 3

    return conv6

def discriminator(images, is_training=True, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        
        if reuse:
            scope.reuse_variables()
 
        conv1 = conv2d(images, output_dim=64, kernel=7, stride=1)
        conv1 = lrelu(conv1)
        #128 x 128 x 64
        
        conv2 = conv2d(conv1, output_dim=64, kernel=7, stride=2)
        conv2 = lrelu(conv2)
        #64 x 64 x 64
            
        conv3 = conv2d(conv2, output_dim=32, stride=2)
        conv3 = lrelu(conv3)
        #32 x 32 x 32

        conv4 = conv2d(conv3, output_dim=1, stride=2)
        conv4 = lrelu(conv4)
        #16 x 16 x 1

        out = dense(conv4 , 1)
    
    return out


def costs_and_vars(real, generated, real_disc, gener_disc, is_training=True):
    '''Return generative and discriminator networks\' costs,
    and variables to optimize them if is_training=True.'''
    d_real_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_disc, \
            labels=tf.ones_like(real_disc)))
    d_gen_cost  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gener_disc, \
            labels=tf.zeros_like(gener_disc)))
     
    g_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gener_disc, \
            labels=tf.ones_like(gener_disc))) * 0.1 + \
            tf.reduce_mean(tf.abs(tf.subtract(generated, real)))

    d_cost = d_real_cost + d_gen_cost
    
    if is_training:
        t_vars = tf.trainable_variables()
        
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]
    
        return g_cost, d_cost, g_vars, d_vars

    else:
        return g_cost, d_cost
