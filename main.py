import os
import scipy.misc
import numpy as np
from model import DCGAN, doresize
from utils import pp, visualize, to_json, get_image, inverse_transform, imread
import tensorflow as tf
from glob import glob
from PIL import Image


# > Define some values
flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 128, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, y_dim=10,
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)
        else:
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                    dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)
        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            dcgan.load(FLAGS.checkpoint_dir)
    #        self.inputs = tf.placeholder(tf.float32,
    #                                    [self.batch_size, self.input_size, self.input_size, 3],
    #                                    name='real_images')
    #        self.up_inputs = tf.image.resize_images(self.inputs, [self.image_shape[0], self.image_shape[1]], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #        self.G = dcgan.generator(z)
    #        self.G_sum = tf.summary.image("G", self.G)
            config = FLAGS
            sample_size = 64
            data = sorted(glob(os.path.join("./data", config.dataset, "valid", "*.jpg")))
            print(data)
            sample_files = data[0:sample_size]
            FLAGS.is_crop = True
            sample = [get_image(sample_file, FLAGS.image_size, is_crop=FLAGS.is_crop) for sample_file in sample_files]
        #    img = Image.fromarray(a[0][1,:,:,:], 'RGB')
        #    img.show()
            img = Image.fromarray(np.uint8(imread(sample_files[0])), 'RGB')
            img.show()
            print(sample[0].shape)
            img = Image.fromarray(np.uint8(inverse_transform(sample[0])*255), 'RGB')
            img.show()
            sample_inputs = [doresize(xx, [32,]*2) for xx in sample]
            img = Image.fromarray(np.uint8(inverse_transform(sample_inputs[0])), 'RGB')
            img.show()
        #    sample_images = np.array(sample).astype(np.float32)
            sample_input_images = np.array(sample_inputs).astype(np.float32)
            a = sess.run([dcgan.G], feed_dict={dcgan.inputs: sample_input_images})
        ##    jpeg_image = PIL.Image.open(a[0])
        #    plt.imshow(jpeg_image)
            print(a[0].shape)
            img = Image.fromarray(np.uint8(((a[0][0,:,:,:])+1.0)*127.5), 'RGB')
            img.show()
    #        return a

        if FLAGS.visualize:
            to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
                                          [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
                                          [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
                                          [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
                                          [dcgan.h4_w, dcgan.h4_b, None])
            # Below is codes for visualization
            OPTION = 2
            visualize(sess, dcgan, FLAGS, OPTION)
        #if FLAGS.test:


if __name__ == '__main__':
    tf.app.run()
