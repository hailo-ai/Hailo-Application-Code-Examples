import numpy as np
import torch
import spconv
import argparse
import glob
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

""" Re-write / create some functions wherever library is not modular enough...
"""

class DemoDataset(DatasetTemplate):
    """ Copied from OpenPCDet/tools/demo.py - 
       - no change in this case, just to collect all utils in one place
    """
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def my_draw_scenes(points, V=None, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None, fig=None,):
    """
        Modified the function from openpcdet/tools/visualization_utils.py for better control
        (e.g., control the figure to which it renders the scene)
    """
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    fig = V.visualize_pts(points, fig=fig)
    fig = V.draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = V.boxes_to_corners_3d(gt_boxes)
        fig = V.draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = V.boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            fig = V.draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(V.box_colormap[k % len(V.box_colormap)])
                mask = (ref_labels == k)
                fig = V.draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100)
    return fig


def load_data_to_CPU(batch_dict):
    """ like original load_data_to_gpu, replaced .cuda() by .cpu()
    """
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cpu().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).cpu().int()
        else:
            batch_dict[key] = torch.from_numpy(val).cpu().float()        
    
    
class PointPillarsCPU(torch.nn.Module):
    """ Wrapper forcing maximal inference on CPU, as a 1st step towards low-end platforms.
        Currently, almost the whole model except NMS postproc which unfortunately is hardcoded cuda.
    """
    def __init__(self, model):
        super().__init__()
        model.cpu()
        # Library creates the anchors in cuda by defalt (applying .cuda() in internal implementation)
        model.dense_head.anchors = [anc.cpu() for anc in model.dense_head.anchors]
        self._pp_model = model

    def forward(self, data_dict):
        load_data_to_CPU(data_dict)
        
        for cur_module in self._pp_model.module_list:
            data_dict = cur_module(data_dict)

        # Here's the unavoidable cuda part:
        return self._pp_model.post_processing({k: (v.cuda() if type(v)==torch.Tensor else v) for k,v in data_dict.items()})
     

class Bev_w_Head(torch.nn.Module):
    """ Same as backbone_2d + head, but accepting spatial_features directly.
         Wraps the original module which it accepts in constructor, code is copied from orig forward().
    """
    def __init__(self, bb2d, dense):
        super().__init__()
        self._bb2d=bb2d  # the model.backbone_2d
        self.dense=dense
        
    def forward(self, spatial_features):
        ups = []
        ret_dict = {}
        x = spatial_features
        # print('x.shape', x.shape)
        for i in range(len(self._bb2d.blocks)):
            x = self._bb2d.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self._bb2d.deblocks) > 0:
                ups.append(self._bb2d.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self._bb2d.deblocks) > len(self._bb2d.blocks):
            x = self._bb2d.deblocks[-1](x)
        
        spatial_features_2d = x
        
        cls_preds = self.dense.conv_cls(spatial_features_2d)
        box_preds = self.dense.conv_box(spatial_features_2d)
        dir_cls_preds = self.dense.conv_dir_cls(spatial_features_2d)
        
        return (spatial_features_2d, cls_preds, box_preds, dir_cls_preds)


class PP_Pre_Bev_w_Head(torch.nn.Module):
    """ This includes the 3D parts which in PointPillars comprise PillarFeatureExtractor+scatter.
        Does NOT include the preceding pre-processing of point cloud + pillar coordinates, 
          into positionally encoded per-pillar 'subclouds' (..as far as i get the code..)
    """
    def __init__(self, pp_full_model):
        super().__init__()
        self.pp_full_model = pp_full_model           
    
    def forward(self, batch_dict):
        for cur_module in self.pp_full_model.module_list[:2]:
            batch_dict = cur_module(batch_dict)
        
        return batch_dict


class PP_Post_Bev_w_Head(torch.nn.Module):
    """ This includes the non-neural anchor-base box-decoding piece of dense head module ("generate_predicted_boxes"),
         as well as the model's postprocessing (using 3D NMS).
    """
    def __init__(self, pp_full_model):
        super().__init__()
        self.pp_full_model = pp_full_model           
    
    def forward(self, bev_out):
        
        spatial_features_2d, cls_preds, box_preds, dir_cls_preds = bev_out # self._hailo_model(spatial_features_hailoinp)

        print(cls_preds.shape, type(cls_preds), box_preds.shape)
        cls_preds = torch.Tensor(cls_preds)
        box_preds = torch.Tensor(box_preds)
        dir_cls_preds = torch.Tensor(dir_cls_preds)
        data_dict = {'batch_size': 1}        
        data_dict['spatial_features_2d'] = torch.Tensor(spatial_features_2d)
        
        batch_cls_preds, batch_box_preds = self.pp_full_model.dense_head.generate_predicted_boxes(
            batch_size=data_dict['batch_size'], cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
        )
        data_dict['batch_cls_preds'] = batch_cls_preds
        data_dict['batch_box_preds'] = batch_box_preds
        data_dict['cls_preds_normalized'] = False
        
        # Here's the unavoidable cuda part:    
        cuda_data_dict = {k: (v.cuda() if type(v)==torch.Tensor else v) for k,v in data_dict.items()}
        pred_dicts, recall_dicts = self.pp_full_model.post_processing(cuda_data_dict)
        return pred_dicts, recall_dicts