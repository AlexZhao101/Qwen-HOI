"""
Test a model and compute detection mAP

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os

import numpy as np
import torch
import argparse
import torchvision
from torch.utils.data import DataLoader

import pocket

from hicodet.hicodet import HICODet
from models import VIPLO
from utils import DataFactory, custom_collate, test

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset


from hicodet.hicodet import HICODet

import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

def recover_boxes(boxes, size):
    boxes = box_cxcywh_to_xyxy(boxes)
    h, w = size
    scale_fct = torch.stack([w, h, w, h])
    boxes = boxes * scale_fct
    return boxes

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def main(args):
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False

    num_anno = torch.tensor(HICODet(None, anno_file=os.path.join(
        args.data_root, 'pvic_instances_train2015.json')).anno_interaction)
    rare = torch.nonzero(num_anno < 10).squeeze(1)
    non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
    num_classes = args.num_class
    dataloader = DataLoader(
        dataset=DataFactory(
            name='hicodet', partition=args.partition,
            data_root=args.data_root,
            detection_root=args.detection_dir, backbone_name=args.backbone_name, num_classes=args.num_class, pose=not args.poseoff
        ), collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True
    )
    object_to_target = dataloader.dataset.dataset.object_to_verb
    object_n_verb_to_interaction = dataloader.dataset.dataset.object_n_verb_to_interaction
    object_to_interaction = dataloader.dataset.dataset.object_to_interaction
    verb_list = dataloader.dataset.dataset.verbs
    net = VIPLO(
        object_to_target, object_n_verb_to_interaction, object_to_interaction, verb_list, 49, num_classes = num_classes, backbone_name=args.backbone_name,
        output_size=args.roi_size, num_iterations=args.num_iter, max_human=args.max_human, max_object=args.max_object,
        box_score_thresh=args.box_score_thresh, patch_size=args.patch_size, pose=not args.poseoff
    )

    epoch = 0
    if os.path.exists(args.model_path):
        print("Loading model from ", args.model_path)
        checkpoint = torch.load(args.model_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint["epoch"]
    elif len(args.model_path):
        print("\nWARNING: The given model path does not exist. "
            "Proceed to use a randomly initialised model.\n")

    net.cuda()
    # test_hico(dataloader,net)
    timer = pocket.utils.HandyTimer(maxlen=1)

    with timer:
        test_ap = test(net, dataloader)
    print("Model at epoch: {} | time elapsed: {:.2f}s\n"
        "Full: {:.4f}, rare: {:.4f}, non-rare: {:.4f}".format(
        epoch, timer[0], test_ap.mean(),
        test_ap[rare].mean(), test_ap[non_rare].mean()
    ))


@torch.no_grad()
def test_hico(dataloader,net):
    net.eval()

    dataset = dataloader.dataset.dataset
    associate = BoxPairAssociation(min_iou=0.5)
    conversion = torch.from_numpy(np.asarray(
        dataset.object_n_verb_to_interaction, dtype=float
    ))


    meter = DetectionAPMeter(
        600, nproc=1, algorithm='11P',
        num_gt=dataset.anno_interaction,
    )
    for batch in tqdm(dataloader):
        # mark有问题
        inputs = pocket.ops.relocate_to_cuda(batch[:-1])
        outputs = net(*inputs)
        outputs = pocket.ops.relocate_to_cpu(outputs, ignore=True)
        targets = batch[-1]
        # contact_masks = batch[-1]

        scores_clt = [];
        preds_clt = [];
        labels_clt = []
        for output, target in zip(outputs, targets):
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
            scores = output['scores']
            verbs = output['labels']
            objects = output['objects']
            interactions = conversion[objects, verbs]
            # Recover target box scale
            gt_bx_h = recover_boxes(target['boxes_h'], target['size'])
            gt_bx_o = recover_boxes(target['boxes_o'], target['size'])

            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(scores)
            unique_hoi = interactions.unique()
            for hoi_idx in unique_hoi:
                gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                if len(gt_idx):
                    labels[det_idx] = associate(
                        (gt_bx_h[gt_idx].view(-1, 4),
                         gt_bx_o[gt_idx].view(-1, 4)),
                        (boxes_h[det_idx].view(-1, 4),
                         boxes_o[det_idx].view(-1, 4)),
                        scores[det_idx].view(-1)
                    )

            scores_clt.append(scores)
            preds_clt.append(interactions)
            labels_clt.append(labels)
        # Collate results into one tensor
        scores_clt = torch.cat(scores_clt)
        preds_clt = torch.cat(preds_clt)
        labels_clt = torch.cat(labels_clt)
        # Gather data from all processes
        scores_ddp = pocket.utils.all_gather(scores_clt)
        preds_ddp = pocket.utils.all_gather(preds_clt)
        labels_ddp = pocket.utils.all_gather(labels_clt)


        meter.append(torch.cat(scores_ddp), torch.cat(preds_ddp), torch.cat(labels_ddp))

    ap = meter.eval()
    return ap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--data-root', default='hicodet', type=str)
    parser.add_argument('--detection-dir', default='hicodet/detections/test2015_gt_vitpose',
                        type=str, help="Directory where detection files are stored")
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--max-human', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--model-path', default='', type=str)
    parser.add_argument('--backbone-name', default='CLIP_CLS', type=str)
    parser.add_argument('--num-class', default=117, type=int)
    parser.add_argument('--patch-size', default=16, type=int)
    parser.add_argument('--roi-size', default=7, type=int)
    parser.add_argument('--poseoff', action='store_true')
    
    args = parser.parse_args()
    print(args)

    main(args)


