# -*- codeing = utf-8 -*-
# @Time: 2022/3/21 0:35
# @Author:xpwang
# @File:demo.py
# @Software:PyCharm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import math
import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time
from grad_cam.utils import GradCAM, show_cam_on_image
import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# SKELETON = [
#     [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
# ]
SKELETON = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [0, 2], [2, 4], [4, 6], [6, 8], [1, 3], [3, 5], [5, 7], [7, 9]]
CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# NUM_KPTS = 17
NUM_KPTS = 10
CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    keypoints_clolr = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 165, 255]]
    assert keypoints.shape == (NUM_KPTS, 2)
    wang = []
    # for i in range(len(SKELETON)):
    #     kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
    #     x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
    #     x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
    #     cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
    #     cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
    #     cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)
    for q, i in enumerate(range(len(SKELETON))):
        if q < 5 :
            kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
            x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
            x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
            neck_x, neck_y = (x_a + x_b) / 2, (y_a + y_b) / 2
            wang.append(neck_x)
            wang.append(neck_y)
            cv2.circle(img, (int(x_a), int(y_a)), 8, keypoints_clolr[i], -1)
            cv2.circle(img, (int(x_b), int(y_b)), 8, keypoints_clolr[i], -1)
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)),     [255, 191, 0], 2)
    #         # TODO 此处我将CocoColors[i]替换为 0,0,0
        else:
            kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
            x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
            x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
            # cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
            # cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)),  [255, 191, 0], 2)

def draw_bbox(box, img):
    """draw the detected bounding box on the image.
    :param img:
    """
    box[0] = int(box[0][0]), int(box[0][1])
    box[1] = int(box[1][0]), int(box[1][1])
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0), thickness=3)


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_classes = pred_classes[:pred_t + 1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    # target_layers = [pose_model.module.swin_transformer.conTran]  # TODO
    # cam = GradCAM(model=pose_model, target_layers=target_layers, use_cuda=False)
    # grayscale_cam = cam(input_tensor=model_input , target_category=target_category)
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        # 1,10,96,72


        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default=r"..\experiments\coco\tokenpose\tokenpose_L_D24_384_288_patch64_dim192_depth24_heads12.yaml")
    parser.add_argument('--video', type=str)
    parser.add_argument('--webcam', action='store_true')
    parser.add_argument('--image', type=str, default="000000001040.jpg")
    parser.add_argument('--write',default=True)
    parser.add_argument('--showFps', action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()

    pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    pose_model.load_state_dict(torch.load(r"F:\pythonProject\move_file\TokenPose-main\tools\output\coco\pose_tokenpose_l\tokenpose_L_D24_384_288_patch64_dim192_depth24_heads12\model_best_4_48_96.pth", map_location=torch.device('cpu')), strict=False)
    # if cfg.TEST.MODEL_FILE:
    #     print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    #     pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device('cpu')), strict=False)
    # else:
    #     print('expected model defined in config at TEST.MODEL_FILE')

    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()

    # Loading an video or an image or webcam
    if args.webcam:
        vidcap = cv2.VideoCapture(0)
    elif args.video:
        vidcap = cv2.VideoCapture(args.video)
    elif args.image:
        image_bgr = cv2.imread(args.image)
    else:
        print('please use --video or --webcam or --image to define the input.')
        return

    if args.webcam or args.video:
        if args.write:
            save_path = 'output.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(save_path, fourcc, 24.0, (int(vidcap.get(3)), int(vidcap.get(4))))
        while True:
            ret, image_bgr = vidcap.read()
            if ret:
                last_time = time.time()
                image = image_bgr[:, :, [2, 1, 0]]

                input = []
                img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(CTX)
                input.append(img_tensor)

                # object detection box
                pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)

                # pose estimation
                if len(pred_boxes) >= 1:
                    for box in pred_boxes:
                        center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                        image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                        pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                        if len(pose_preds) >= 1:
                            for kpt in pose_preds:
                                draw_pose(kpt, image_bgr)  # draw the poses

                if args.showFps:
                    fps = 1 / (time.time() - last_time)
                    img = cv2.putText(image_bgr, 'fps: ' + "%.2f" % (fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 255, 0), 2)

                if args.write:
                    out.write(image_bgr)

                cv2.imshow('demo', image_bgr)
                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break
            else:
                print('cannot load the video.')
                break

        cv2.destroyAllWindows()
        vidcap.release()
        if args.write:
            print('video has been saved as {}'.format(save_path))
            out.release()

    else:
        # estimate on the image
        last_time = time.time()
        image = image_bgr[:, :, [2, 1, 0]]
        input = []
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # CHW
        img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(CTX)
        input.append(img_tensor)
        # object detection box
        # pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)
        # pred_boxes =[[[728.6140350877192,319.12280701754383],[1453.1754385964914,650.7017543859645]],
        #              [[932.1228070175438,508.59649122807014],[1765.456140350877,908.5964912280701]],
        #              [[242.64912280701753,312.1052631578947],[760.1929824561403,587.5438596491227]]]
        # pred_boxes= [[[1039.0625,431.75],[1426.5625, 1145.8125]],[[1312.5,572.375],[1964.0625, 895.8125]]] # w1

        """
        [
            [
                1353.125,
                192.6875
            ],
            [
                1989.0625,
                691.125
            ]
        ],
        [
            [
                1385.9375,
                0
            ],
            [
                2053.125,
                255.1875
            ]
        ],
        """
        # [
        #     [
        #         99.0,
        #         112.29479768786128
        #     ],
        #     [
        #         736.5722543352601,
        #         464.3179190751445
        #     ]
        # ]
        pred_boxes = [[
            [
                99.0,
                112.29479768786128
            ],
            [
                736.5722543352601,
                464.3179190751445
            ]
        ]]
        # ], [
        #     [
        #         489.0625,
        #         402.0625
        #     ],
        #     [
        #         1098.4375,
        #         847.375
        #     ]
        # ],
        #     [
        #         [
        #             420.3125,
        #             131.75
        #         ],
        #         [
        #             1059.375,
        #             445.8125
        #         ]
        #     ]
        # ]
        # pose estimation
        if len(pred_boxes) >= 1:
            for box in pred_boxes:
                box[0] = int(box[0][0]), int(box[0][1])
                box[1] = int(box[1][0]), int(box[1][1])
                picture = cv2.rectangle(image, box[0], box[1], color=(0, 255, 0), thickness=3)
                # cv2.imshow("fff", picture)
                # cv2.waitKey(0)
                center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                # cv2.rectangle(img, box[0], box[1], color=(0, 255, 0), thickness=3)
                if len(pose_preds) >= 1:
                    for kpt in pose_preds:
                        shoulder_mid = abs(kpt[0] + kpt[1]) / 2
                        tail_mid = abs(kpt[8] + kpt[9]) / 2
                        body_length = math.sqrt(
                            math.pow(abs(shoulder_mid[0] - tail_mid[0]), 2) + math.pow(
                                abs(shoulder_mid[1] - tail_mid[1]), 2))
                        body_width = math.sqrt(
                            math.pow(abs(kpt[2] - kpt[3])[0], 2) + math.pow(abs(kpt[2] - kpt[3])[1], 2))
                        belly_width = math.sqrt(
                            math.pow(abs(kpt[4] - kpt[5])[0], 2) + math.pow(abs(kpt[4] - kpt[5])[1], 2))
                        hip_width = math.sqrt(
                            math.pow(abs(kpt[6] - kpt[7])[0], 2) + math.pow(abs(kpt[6] - kpt[7])[1], 2))
                        # if i == 0:
                        # print(tuple(kpt[0]+10))
                        a = "%.2f;%.2f;%.2f;%.2f;" % (
                        body_length * 1.93, body_width * 1.93, belly_width * 1.93, hip_width * 1.93)
                        # cv2.putText(image_bgr, a, tuple([int(i) for i in (kpt[4] - 18)]), cv2.FONT_HERSHEY_SIMPLEX,
                        #             0.75, (0, 0, 255), 2)
                        draw_pose(kpt, image_bgr)  # draw the poses
                        cv2.imwrite("final.jpg", image_bgr)
        if args.showFps:
            fps = 1 / (time.time() - last_time)
            img = cv2.putText(image_bgr, 'fps: ' + "%.2f" % (fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0),
                              2)

        if args.write:
            save_path = 'output.jpg'
            cv2.imwrite(save_path, image_bgr)
            print('the result image has been saved as {}'.format(save_path))

        cv2.imshow('demo', image_bgr)
        if cv2.waitKey(0) & 0XFF == ord('q'):
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
