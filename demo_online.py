# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os

import numpy as np
import tensorflow as tf

from config import *
from train import _draw_box
from nets import *
import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint', '../data_test_online/models/train/model.ckpt-49999',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def cam_demo(outDir):
    """Detect image."""

    assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
        'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)


    with tf.Graph().as_default():
        # Load model
        if FLAGS.demo_net == 'squeezeDet':
            mc = kitti_squeezeDet_config()
            mc.BATCH_SIZE = 1
            # model parameters will be restored from checkpoint
            mc.LOAD_PRETRAINED_MODEL = False
            model = SqueezeDet(mc, FLAGS.gpu)
        elif FLAGS.demo_net == 'squeezeDet+':
            mc = kitti_squeezeDetPlus_config()
            mc.BATCH_SIZE = 1
            mc.LOAD_PRETRAINED_MODEL = False
            model = SqueezeDetPlus(mc, FLAGS.gpu)

        saver = tf.train.Saver(model.model_params)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver.restore(sess, FLAGS.checkpoint)
            idx_frame = 1
            # for f in sorted(glob.iglob(FLAGS.input_path)):
            cam = cv2.VideoCapture(0)
            while True:
                ret_val, img1 = cam.read()
                img1 = rotate_bound(img1, 90)
                # img1 = cv2.imread(f)
                im = img1.astype(np.float32, copy=True)

                im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
                img = cv2.resize(img1, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
                input_image = im - mc.BGR_MEANS

                # Detect
                det_boxes, det_probs, det_class = sess.run(
                    [model.det_boxes, model.det_probs, model.det_class],
                    feed_dict={model.image_input: [input_image]})

                # Filter
                final_boxes, final_probs, final_class = model.filter_prediction(
                    det_boxes[0], det_probs[0], det_class[0])

                keep_idx = [idx for idx in range(len(final_probs)) \
                            if final_probs[idx] > 0.75]  # mc.PLOT_PROB_THRESH
                final_boxes = [final_boxes[idx] for idx in keep_idx]
                final_probs = [final_probs[idx] for idx in keep_idx]
                final_class = [final_class[idx] for idx in keep_idx]

                # TODO(bichen): move this color dict to configuration file
                cls2clr = {
                    '01ball': (255, 191, 0),
                    '02basket': (0, 191, 255),
                    # 'pedestrian':(255, 0, 191)
                }

                # Draw boxes
                _draw_box(
                    img, final_boxes,
                    [mc.CLASS_NAMES[idx] + ': (%.2f)' % prob \
                     for idx, prob in zip(final_class, final_probs)],
                    cdict=cls2clr,
                )

                if not os.path.isdir(os.path.join(outDir, 'original_frames')):
                    os.makedirs(os.path.join(outDir, 'original_frames'))
                if not os.path.isdir(os.path.join(outDir, 'result_frames')):
                    os.makedirs(os.path.join(outDir, 'result_frames'))

                cv2.imwrite(os.path.join(outDir, 'original_frames', '%06d.jpg') % idx_frame, img1)
                cv2.imwrite(os.path.join(outDir, 'result_frames', '%06d.jpg') % idx_frame, img)

                idx_frame += 1
                cv2.imshow('my webcam', img)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
            cv2.destroyAllWindows()


def main(argv=None):

    now = datetime.datetime.now()
    newDirName = '../data_test_online/' + now.strftime("%Y_%m_%d-%H%M%S")

    # os.mkdir(newDirName)
    #
    # outDirs = '../data_test_online/' + today.strftime('%Y%m%d') + h
    tf.gfile.MakeDirs(newDirName)
    cam_demo(newDirName)


if __name__ == '__main__':
    tf.app.run()
