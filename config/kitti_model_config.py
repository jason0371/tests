# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for pascal dataset"""

import numpy as np

from .config import base_model_config

def kitti_model_config():
  """Specify the parameters to tune below."""
  mc                       = base_model_config('KITTI')
  # mc.IMAGE_WIDTH           = 1864 # half width 621
  # mc.IMAGE_HEIGHT          = 562 # half height 187
  mc.IMAGE_WIDTH           = 640#1248 # half width 621
  mc.IMAGE_HEIGHT          = 360#384 # half height 187
  # mc.IMAGE_WIDTH           = 621
  # mc.IMAGE_HEIGHT          = 187

  mc.WEIGHT_DECAY          = 0.0001
  mc.PROB_THRESH           = 0.005
  mc.TOP_N_DETECTION       = 64
  mc.PLOT_PROB_THRESH      = 0.4
  mc.NMS_THRESH            = 0.4
  mc.LEARNING_RATE         = 0.01
  mc.MOMENTUM              = 0.9
  mc.DECAY_STEPS           = 10000
  mc.LR_DECAY_FACTOR       = 0.5
  mc.BATCH_SIZE            = 20
  mc.LOSS_COEF_BBOX        = 5.0
  mc.LOSS_COEF_CONF_POS    = 75.0
  mc.LOSS_COEF_CONF_NEG    = 100.0
  mc.LOSS_COEF_CLASS       = 1.0
  mc.MAX_GRAD_NORM         = 1.0
  mc.DATA_AUGMENTATION     = True
  mc.DRIFT_X               = 150
  mc.DRIFT_Y               = 100
  mc.ANCHOR_BOX            = set_anchors(mc)
  mc.ANCHORS               = len(mc.ANCHOR_BOX)
  mc.ANCHOR_PER_GRID       = 4
  mc.USE_DECONV            = False
  mc.EXCLUDE_HARD_EXAMPLES = False

  return mc

def set_anchors(mc):

  H, W, B = 23, 40, 4
  anchor_shapes = np.reshape(
      [np.array(
          [[mc.IMAGE_HEIGHT / (384 / 366.), mc.IMAGE_WIDTH / (1248 / 174.)],
           [mc.IMAGE_HEIGHT / (384 / 162.), mc.IMAGE_WIDTH / (1248 / 87.)],
           [mc.IMAGE_HEIGHT / (384 / 258.), mc.IMAGE_WIDTH / (1248 / 173.)],
           [mc.IMAGE_HEIGHT / (384 / 78.), mc.IMAGE_WIDTH / (1248 / 170.)]])] * H * W,
      (H, W, B, 2)
  )

  # H, W, B = 30, 15, 9  # 24, 78, 9
  # anchor_shapes = np.reshape(
  #     [np.array(
  #         [[480 / (1248 / 36.), 240 / (384 / 37.)], [480 / (1248 / 366.), 240 / (384 / 174.)],
  #          [480 / (1248 / 115.), 240 / (384 / 59.)],
  #          [480 / (1248 / 162.), 240 / (384 / 87.)], [480 / (1248 / 38.), 240 / (384 / 90.)],
  #          [480 / (1248 / 258.), 240 / (384 / 173.)],
  #          [480 / (1248 / 224.), 240 / (384 / 108.)], [480 / (1248 / 78.), 240 / (384 / 170.)],
  #          [480 / (1248 / 72.), 240 / (384 / 43.)]])] * H * W,
  #     (H, W, B, 2)
  # )

  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(mc.IMAGE_WIDTH)/(W+1)]*H*B), 
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(mc.IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors
