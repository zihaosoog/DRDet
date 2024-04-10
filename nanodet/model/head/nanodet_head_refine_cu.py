# Copyright 2021 RangiLyu.
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

import torch,math
import torch.nn as nn
from nanodet.util import *
from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from .nanodet_plus_head import NanoDetPlusHead
from ..module.refine import *
from .gfl_head import *
from ..loss.iou_loss import *
class NanoDetHeadRFCU(NanoDetPlusHead):
    """
    Modified from GFL, use same loss functions but much lightweight convolution heads
    """

    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        strides=[8, 16, 32],
        conv_type="DWConv",
        norm_cfg=dict(type="BN"),
        reg_max=7,
        activation="LeakyReLU",
        assigner_cfg=dict(topk=13),
        **kwargs
    ):
        self.share_cls_reg = False
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule
        super(NanoDetHeadRFCU, self).__init__(
            num_classes,
            loss,
            input_channel,
            feat_channels,
            stacked_convs,
            kernel_size,
            strides,
            conv_type,
            norm_cfg,
            reg_max,
            activation,
            assigner_cfg,
            **kwargs
        )
        self.loss_dn = nn.SmoothL1Loss(reduction = 'sum')
    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.refine_boxes = nn.ModuleList()
        for stride in self.strides:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            refine_box = Refinebox(in_channel=self.feat_channels,stride=stride,reg_max=self.reg_max)
            self.refine_boxes.append(refine_box)

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    self.num_classes + 4 * (self.reg_max + 1)
                    if self.share_cls_reg
                    else self.num_classes,
                    1,
                    padding=0,
                )
                for _ in self.strides
            ]
        )
        # TODO: if
        self.gfl_reg = nn.ModuleList(
            [
                nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 1, padding=0)
                for _ in self.strides
            ]
        )

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    5,
                    stride=1,
                    padding=2,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
            if not self.share_cls_reg:
                reg_convs.append(
                    self.ConvModule(
                        chn,
                        self.feat_channels,
                        5,
                        stride=1,
                        padding=2,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None,
                        activation=self.activation,
                    )
                )

        return cls_convs, reg_convs

    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)
        print("Finish initialize NanoDet Head.")

    def forward(self, feats):
        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(feats)
        outputs = []
        for x, cls_convs, reg_convs, gfl_cls, gfl_reg, refine_box in zip(
            feats, self.cls_convs, self.reg_convs, self.gfl_cls, self.gfl_reg, self.refine_boxes
        ):
            cls_feat = x
            reg_feat = x
            for cls_conv in cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)
            if self.share_cls_reg:
                output = gfl_cls(cls_feat)
            else:
                cls_score = gfl_cls(cls_feat)
                bbox_pred_init = gfl_reg(reg_feat)

                b,_,w,h = bbox_pred_init.shape
                bbox_preds, bbox_pred_refines = [],[]

                for i in range(self.reg_max + 1):

                    bbox_pred,bbox_pred_refine = refine_box(pre_box=bbox_pred_init.reshape(b,-1,4,w,h)[:,i,...],reg_feat=reg_feat)
                    bbox_preds.append(bbox_pred)
                    bbox_pred_refines.append(bbox_pred_refine)

                bbox_preds = torch.cat(bbox_preds,dim=1)
                bbox_pred_refines = torch.cat(bbox_pred_refines,dim=1)

                output = torch.cat([cls_score, bbox_preds, bbox_pred_refines], dim=1)
            outputs.append(output.flatten(start_dim=2))
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs

    def _forward_onnx(self, feats):
        """only used for onnx export"""
        outputs = []
        for x, cls_convs, reg_convs, gfl_cls, gfl_reg, refine_box in zip(
            feats, self.cls_convs, self.reg_convs, self.gfl_cls, self.gfl_reg, self.refine_boxes
        ):
            cls_feat = x
            reg_feat = x
            for cls_conv in cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)
            if self.share_cls_reg:
                output = gfl_cls(cls_feat)
                cls_score, bbox_pred = output.split(
                    [self.num_classes, 4 * (self.reg_max + 1)], dim=1
                )
            else:
                cls_score = gfl_cls(cls_feat)
                bbox_pred = gfl_reg(reg_feat)

                b,_,w,h = bbox_pred.shape
                bbox_preds, bbox_pred_refines = [],[]
                
                for i in range(self.reg_max + 1):
                    bbox_pred,bbox_pred_refine = refine_box(pre_box=bbox_pred.reshape(b,-1,4,w,h)[:,i,...],reg_feat=reg_feat)
                    bbox_preds.append(bbox_pred)
                    bbox_pred_refines.append(bbox_pred_refine)

                bbox_preds = torch.cat(bbox_preds,dim=1)
                bbox_pred_refines = torch.cat(bbox_pred_refines,dim=1)

                output = torch.cat([cls_score, bbox_preds, bbox_pred_refines], dim=1)
            outputs.append(output.flatten(start_dim=2))
        return torch.cat(outputs, dim=2).permute(0, 2, 1)

    def loss(self, preds, gt_meta, aux_preds=None,epoch= 0 ):
        """Compute losses.
        Args:
            preds (Tensor): Prediction output.
            gt_meta (dict): Ground truth information.
            aux_preds (tuple[Tensor], optional): Auxiliary head prediction output.

        Returns:
            loss (Tensor): Loss tensor.
            loss_states (dict): State dict of each loss.
        """
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_labels = gt_meta["gt_labels"]
        device = preds.device
        batch_size = preds.shape[0]
        input_height, input_width = gt_meta["img"].shape[2:]
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)

        cls_preds, reg_preds, reg_refine = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1), 4 * (self.reg_max + 1)], dim=-1
        )
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)
        dis_preds_refine = self.distribution_project(reg_refine) * center_priors[..., 2, None]
        decoded_bboxes_refine = distance2bbox(center_priors[..., :2], dis_preds_refine)
        if aux_preds is not None:
            # use auxiliary head to assign
            aux_cls_preds, aux_reg_preds, aux_reg_preds_refine = aux_preds.split(
                [self.num_classes, 4 * (self.reg_max + 1),4 * (self.reg_max + 1)], dim=-1
            )
            aux_dis_preds = (
                self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            )
            aux_dis_preds_refine = (
                self.distribution_project(aux_reg_preds_refine) * center_priors[..., 2, None]
            )
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)
            aux_decoded_bboxes_refine = distance2bbox(center_priors[..., :2], aux_dis_preds_refine)
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                aux_cls_preds.detach(),
                center_priors,
                aux_decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
            )
        else:
            # use self prediction to assign
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                cls_preds.detach(),
                center_priors,
                decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
            )

        loss, loss_states = self._get_loss_from_assign(
            cls_preds, reg_preds, decoded_bboxes, decoded_bboxes_refine, batch_assign_res,epoch
        )

        if aux_preds is not None:
            aux_loss, aux_loss_states = self._get_loss_from_assign(
                aux_cls_preds, aux_reg_preds, aux_decoded_bboxes, aux_decoded_bboxes_refine, batch_assign_res,epoch
            )
            loss = loss + aux_loss
            for k, v in aux_loss_states.items():
                loss_states["aux_" + k] = v
        return loss, loss_states

    def _get_loss_from_assign(self, cls_preds, reg_preds, decoded_bboxes, decoded_bboxes_refine, assign, epoch):
        device = cls_preds.device
        labels, label_scores, bbox_targets, dist_targets, num_pos = assign
        num_total_samples = max(
            reduce_mean(torch.tensor(sum(num_pos)).to(device)).item(), 1.0
        )

        labels = torch.cat(labels, dim=0)
        label_scores = torch.cat(label_scores, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        decoded_bboxes_refine = decoded_bboxes_refine.reshape(-1, 4)
        loss_qfl = self.loss_qfl(
            cls_preds, (labels, label_scores), avg_factor=num_total_samples
        )

        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False
        ).squeeze(1)

        if len(pos_inds) > 0:
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

            # iou_targets_rf = bbox_overlaps(decoded_bboxes_refine[pos_inds],bbox_targets[pos_inds].detach()).clamp(min=1e-6)
            # bbox_weights_rf = iou_targets_rf.clone().detach()
            # bbox_avg_factor_rf = reduce_mean(
            #     bbox_weights_rf.sum()).clamp_(min=1).item()
            th = (0.1 * (epoch-100)^2 + 18*18) if epoch > 150 else 10000000
            areas = (bbox_targets[pos_inds][:,2] - bbox_targets[pos_inds][:,0]) * (bbox_targets[pos_inds][:,3] - bbox_targets[pos_inds][:,1])
            weight_curr = th /(nn.functional.relu(areas-th)+th)
            bbox_avg_factor_cu = max(reduce_mean(weight_curr.sum()).item(), 1.0)

            loss_bbox_curr = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_curr,
                avg_factor=bbox_avg_factor_cu,
            )

            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )

            loss_bbox_refine = self.loss_bbox(
                decoded_bboxes_refine[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )

            dist_targets = torch.cat(dist_targets, dim=0)
            loss_dfl = self.loss_dfl(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),
                dist_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor,
            )
        else:
            loss_bbox = reg_preds.sum() * 0
            loss_dfl = reg_preds.sum() * 0
            loss_bbox_refine = reg_preds.sum() * 0
            loss_bbox_curr = reg_preds.sum() * 0

        loss = loss_qfl + loss_bbox + loss_dfl + loss_bbox_refine + loss_bbox_curr
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl, loss_bbox_refine=loss_bbox_refine,loss_bbox_curr = loss_bbox_curr)
        return loss, loss_states


    def post_process(self, preds, meta):
        """Prediction results post processing. Decode bboxes and rescale
        to original image size.
        Args:
            preds (Tensor): Prediction output.
            meta (dict): Meta info.
        """
        cls_scores, bbox_preds_wo_refine, bbox_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1), 4 * (self.reg_max + 1)], dim=-1
        )
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        det_results = {}
        warp_matrixes = (
            meta["warp_matrix"]
            if isinstance(meta["warp_matrix"], list)
            else meta["warp_matrix"]
        )
        img_heights = (
            meta["img_info"]["height"].cpu().numpy()
            if isinstance(meta["img_info"]["height"], torch.Tensor)
            else meta["img_info"]["height"]
        )
        img_widths = (
            meta["img_info"]["width"].cpu().numpy()
            if isinstance(meta["img_info"]["width"], torch.Tensor)
            else meta["img_info"]["width"]
        )
        img_ids = (
            meta["img_info"]["id"].cpu().numpy()
            if isinstance(meta["img_info"]["id"], torch.Tensor)
            else meta["img_info"]["id"]
        )

        for result, img_width, img_height, img_id, warp_matrix in zip(
            result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            det_result = {}
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(
                det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
            )
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            det_results[img_id] = det_result
        return det_results
