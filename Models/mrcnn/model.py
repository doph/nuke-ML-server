# Copyright (c) 2019 Foundry.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

# The inference method is based on:
# --------------------------------------------------------
# Facebook infer_simple.py file:
# https://github.com/facebookresearch/Detectron/blob/master/tools/infer_simple.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Apache License, Version 2.0
# --------------------------------------------------------

import copy
import numpy as np

from caffe2.python import workspace
# import libcaffe2_detectron_ops_gpu.so
import detectron.utils.c2 as c2_utils
c2_utils.import_detectron_ops()

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file, merge_cfg_from_cfg
from detectron.utils.collections import AttrDict
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils

from .vis import vis_one_image_binary, vis_one_image_opencv
from .utils import dict_equal
from ..common.util import linear_to_srgb, srgb_to_linear
from ..baseModel import BaseModel

class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()
        self.name = 'Mask RCNN'

        # Configuration and weights options
        # By default, we use ResNet50 backbone architecture, you can switch to
        # ResNet101 to increase quality if your GPU memory is higher than 8GB.
        # To do so, you will need to download both .yaml and .pkl ResNet101 files
        # then replace the below 'cfg_file' with the following:
        # self.cfg_file = 'models/mrcnn/e2e_mask_rcnn_X-101-64x4d-FPN_2x.yaml'
        self.cfg_file = 'models/mrcnn/e2e_mask_rcnn_R-50-FPN_2x.yaml'
        self.weights = 'models/mrcnn/model_final.pkl'
        self.default_cfg = copy.deepcopy(AttrDict(cfg)) # cfg from detectron.core.config
        self.mrcnn_cfg = AttrDict()
        self.dummy_coco_dataset = dummy_datasets.get_coco_dataset()

        # Inference options
        self.show_box = True
        self.show_class = True
        self.thresh = 0.7
        self.alpha = 0.4
        self.show_border = True
        self.border_thick = 1
        self.bbox_thick = 1
        self.font_scale = 0.35
        self.binary_masks = False

        # Define exposed options
        self.options = (
            'show_box', 'show_class', 'thresh', 'alpha', 'show_border',
            'border_thick', 'bbox_thick', 'font_scale', 'binary_masks',
            )
        # Define inputs/outputs
        self.inputs = {'input': 3}
        self.outputs = {'output': 3}

    def inference(self, image_list):
        """Do an inference on the model with a set of inputs.

        # Arguments:
            image_list: The input image list

        Return the result of the inference.
        """
        image = image_list[0]
        image = linear_to_srgb(image)*255.
        imcpy = image.copy()

        # Initialize the model out of the configuration and weights files
        if not hasattr(self, 'model'):
            workspace.ResetWorkspace()
            # Reset to default config
            merge_cfg_from_cfg(self.default_cfg)
            # Load mask rcnn configuration file
            merge_cfg_from_file(self.cfg_file)
            assert_and_infer_cfg(cache_urls=False, make_immutable=False)
            self.model = infer_engine.initialize_model_from_cfg(self.weights)
            # Save mask rcnn full configuration file
            self.mrcnn_cfg = copy.deepcopy(AttrDict(cfg)) # cfg from detectron.core.config
        else:
            # There is a global config file for all detectron models (Densepose, Mask RCNN..)
            # Check if current global config file is correct for mask rcnn
            if not dict_equal(self.mrcnn_cfg, cfg):
                # Free memory of previous workspace
                workspace.ResetWorkspace()
                # Load mask rcnn configuration file
                merge_cfg_from_cfg(self.mrcnn_cfg)
                assert_and_infer_cfg(cache_urls=False, make_immutable=False)
                self.model = infer_engine.initialize_model_from_cfg(self.weights)

        with c2_utils.NamedCudaScope(0):
            # If using densepose/detectron GitHub, im_detect_all also returns cls_bodys
            # Only takes the first 3 elements of the list for compatibility
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                self.model, image[:, :, ::-1], None
                )[:3]

        if self.binary_masks:
            res = vis_one_image_binary(
                imcpy,
                cls_boxes,
                cls_segms,
                thresh=self.thresh
                )
        else:
            res = vis_one_image_opencv(
                imcpy,
                cls_boxes,
                cls_segms,
                cls_keyps,
                thresh=self.thresh,
                show_box=self.show_box,
                show_class=self.show_class,
                dataset=self.dummy_coco_dataset,
                alpha=self.alpha,
                show_border=self.show_border,
                border_thick=self.border_thick,
                bbox_thick=self.bbox_thick,
                font_scale=self.font_scale
                )

        res = srgb_to_linear(res.astype(np.float32) / 255.)

        return [res]