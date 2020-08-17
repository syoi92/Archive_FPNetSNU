from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def resnet51(image, options, reuse=False, name="resnet"):
    with tf.variable_scope(name):
        # image is fine_size x fine_size x input_nc
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, is_proj=False, name='res'):
            if is_proj:
                x = instance_norm(conv2d(x, dim, 1, 1, name=name + '_p'), name + '_bn')  # projcetion
            y = lrelu(instance_norm(conv2d(x, dim / 4, 1, 1, name=name + '_c1'), name + '_bn1'))
            y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
            y = lrelu(instance_norm(conv2d(y, dim / 4, 3, 1, padding='VALID', name=name + '_c2'), name + '_bn2'))
            y = instance_norm(conv2d(y, dim, 1, 1, name=name + '_c3'), name + '_bn3')
            return x + y

        # Modified ResNet51
        # b1 + max pool
        b = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "SYMMETRIC")
        b = lrelu(instance_norm(conv2d(b, options.gf_dim, 7, 2, padding='VALID', name='b1_c'), 'b1_bn'))
        b = tf.nn.max_pool(b, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
        # b2
        for i in range(3):
            is_proj = True if i == 0 else False
            b = residule_block(b, options.gf_dim * 4, is_proj=is_proj, name='b2_' + str(i))
        # b3
        for i in range(4):
            is_proj = True if i == 0 else False
            b = residule_block(b, options.gf_dim * 8, is_proj=is_proj, name='b3_' + str(i))
        # b4
        for i in range(6):
            is_proj = True if i == 0 else False
            b = residule_block(b, options.gf_dim * 16, is_proj=is_proj, name='b4_' + str(i))
        # b5
        for i in range(3):
            is_proj = True if i == 0 else False
            b = residule_block(b, options.gf_dim * 32, is_proj=is_proj, name='b5_' + str(i))

        b6 = conv2d(b, options.gf_dim, 1, 1, name='b6_c')
        b6 = lrelu(instance_norm(b6, 'b6_c_bn'))
        b6 = deconv2d(b6, options.gf_dim, 4, 4, name='b6_dc')
        b6 = lrelu(instance_norm(b6, 'b6_dc_bn'))

        b7 = tf.pad(b6, [[0, 0], [3, 3], [3, 3], [0, 0]], "SYMMETRIC")
        b7 = conv2d(b7, options.output_nc, 7, 1, padding='VALID', name='b7_c')
        return b7


def resnet152(image, options, reuse=False, name="resnet152"):
    with tf.variable_scope(name):
        # image is 256 x 256 x (input_c_dim+output_c_dim)
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, is_first=False, is_proj=False, name='res'):
            if is_proj:
                x = instance_norm(conv2d(x, dim, 1, 1, name=name + '_p'), name + '_bn') #projcetion

            if is_first:
                y = lrelu(instance_norm(conv2d(x, dim/4, 1, 2, name=name + '_c1'), name + '_bn1'))
                x = instance_norm(conv2d(x, dim, 2, 2, name=name + '_p'), name + '_bn')  # projection
            else:
                y = lrelu(instance_norm(conv2d(x, dim/4, 1, 1, name=name + '_c1'), name + '_bn1'))

            y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
            y = lrelu(instance_norm(conv2d(y, dim/4, 3, 1, padding='VALID', name=name+'_c2'), name+'_bn2'))
            y = instance_norm(conv2d(y, dim, 1, 1, name=name + '_c3'), name + '_bn3')
            return x+y

        # Modified ResNet152
        # b1 + max pool
        b1 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "SYMMETRIC")
        b1 = lrelu(instance_norm(conv2d(b1, options.gf_dim, 7, 2, padding='VALID', name='b1_c'), 'b1_bn'))
        b1 = tf.nn.max_pool(b1, [1,3,3,1], [1,2,2,1], padding='SAME')

        # b2
        for i in range(3):
            if i == 0:
                b2 = residule_block(b1, options.gf_dim*4, is_proj=True, name='b2_'+str(i))
            else:
                b2 = residule_block(b2, options.gf_dim*4, name='b2_'+str(i))
        # b3
        for i in range(8):
            if i == 0:
                b3 = residule_block(b2, options.gf_dim*8, is_proj=True, name='b3_'+str(i))
            else:
                b3 = residule_block(b3, options.gf_dim*8, name='b3_'+str(i))
        #b4
        for i in range(36):
            if i == 0:
                b4 = residule_block(b3, options.gf_dim*16, is_proj=True, name='b4_'+str(i))
            else:
                b4 = residule_block(b4, options.gf_dim*16, name='b4_'+str(i))
        #b5
        for i in range(3):
            if i == 0:
                b5 = residule_block(b4, options.gf_dim*32,  is_proj=True, name='b5_'+str(i))
            else:
                b5 = residule_block(b5, options.gf_dim*32,  name='b5_'+str(i))

        b6 = conv2d(b5, options.gf_dim, 1, 1, name='b6_c')
        b6 = lrelu(instance_norm(b6, 'b6_c_bn'))
        b6 = deconv2d(b6, options.gf_dim, 4, 4, name='b6_dc')
        b6 = lrelu(instance_norm(b6, 'b6_dc_bn'))

        b7 = tf.pad(b6, [[0,0], [3,3], [3,3], [0,0]], "SYMMETRIC")
        b7 = conv2d(b7, options.output_nc, 7, 1, padding='VALID', name='b7_c')

        return b7


def discriminator(image, options, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target)) 


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target) ** 2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    #  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    

