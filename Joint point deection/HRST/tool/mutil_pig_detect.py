from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

##########
import os
import time
import json
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, mobilenetv3_fpn, ConvNeXt_fpn, efficv2_fpn
# from draw_box_utils import draw_box
from fvcore.nn import FlopCountAnalysis, parameter_count_table
##########
import os
import json
import time

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

from build_utils import img_utils, torch_utils, utils
from YOLOv4_models import Darknet
from YOLOv4_draw_box_utils import draw_box
# TODO 关键点
import collections

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

import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

SKELETON = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [0, 2], [2, 4], [4, 6], [6, 8], [1, 3], [3, 5], [5, 7],
            [7, 9]]
CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
              [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
NUM_KPTS = 10
def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    wang = []
    assert keypoints.shape == (NUM_KPTS, 2)
    # for i in range(len(SKELETON)):
    #     kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
    #     x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
    #     x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
    #     cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
    #     cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
    #     cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)
    #  TODO 显示几个rgb(238,238,238)
    keypoints_clolr = [[0,0,0], [255,0,0], [0,255,0], [0,0,255], [0, 165, 255]]
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
    two_points_dist = math.sqrt(math.pow(abs(wang[0]-wang[8]),2)+math.pow(abs(wang[1]-wang[9]),2))
    five_dist = math.sqrt(math.pow(abs(wang[0]-wang[2]),2)+ math.pow(abs(wang[1]-wang[3]),2))+\
                 math.sqrt(math.pow(abs(wang[2]-wang[4]),2)+ math.pow(abs(wang[3]-wang[5]),2))+\
                 math.sqrt(math.pow(abs(wang[4] - wang[6]), 2) + math.pow(abs(wang[5] - wang[7]), 2))+ \
                 math.sqrt(math.pow(abs(wang[6] - wang[8]), 2) + math.pow(abs(wang[7] - wang[9]), 2))
    print(two_points_dist/five_dist)


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
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
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
    parser.add_argument('--cfg', type=str,
                        default=r"..\experiments\coco\tokenpose\tokenpose_L_D24_384_288_patch64_dim192_depth24_heads12.yaml")
    parser.add_argument('--video', type=str)
    parser.add_argument('--webcam', action='store_true')
    parser.add_argument('--image', type=str, default="000000001040.jpg")
    parser.add_argument('--write', default=True)
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


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    # backbone = mobilenetv3_fpn.mobilenet_fpn_backbone()
    backbone = ConvNeXt_fpn.ConvNeXt_fpn_backbone()
    # backbone = efficv2_fpn.mobilenet_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    device = "cpu"
    args = parse_args()
    update_config(cfg, args)  # TODO 关键点cfg配置信息
    pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )
    pose_model.load_state_dict(torch.load(
        r"F:\pythonProject\move_file\TokenPose-main\tools\output\coco\pose_tokenpose_l\model_best_4_48_96.pth",
        map_location=torch.device('cpu')), strict=False)
    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(device)
    pose_model.eval()
    model = create_model(num_classes=3)
    # load train weights
    # train_weights = "./log_result/mov3_final.pth"
    # train_weights = "./log_result/resnet.pth"
    # train_weights = "./log_result/C.pth"
    train_weights = "./log_result/C.pth"

    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    # load image
    original_img = Image.open("./q10.jpg")

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # tensor = (torch.rand(1, 3, 720, 1280),)
        # # tensor = (torch.rand(1, 3, 512, 512),)
        # # 分析FLOPs
        # flops = FlopCountAnalysis(model, tensor)
        # print("FLOPs: ", flops.total())
        #
        # # 分析parameters
        # print(parameter_count_table(model))
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        # TODO 如果处于站立状态进行关键点检测
        stand_list = []
        for i, j in enumerate(predict_classes):
            if j == 1:
                stand_list.append(predict_boxes[i, :])
        stand_numpy = np.array(stand_list)  # (3, 4)
        image_pose = cv2.imread("./q10.jpg")
        # image_pose = img_o.copy()
        # TODO  最终的实验结果
        stand_body_length = collections.defaultdict(list)
        twist = collections.defaultdict(list)
        if len(stand_numpy) >= 1:
            for box in stand_numpy:
                box = box.reshape(2, 2)
                center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                if len(pose_preds) >= 1:
                    for kpt in pose_preds:
                        draw_pose(kpt, image_pose)  # draw the poses
            # cv2.imwrite("final.jpg", image_pose)
        if args.write:
            save_path = 'output.jpg'
            cv2.imwrite(save_path, image_pose)
            print('the result image has been saved as {}'.format(save_path))
        img_o = draw_box(image_pose[:, :, ::-1],  predict_boxes, predict_classes, predict_scores, category_index, stand_body_length)
        # plt.imshow(img_o)
        img = cv2.cvtColor(np.asarray(img_o), cv2.COLOR_RGB2BGR)
        cv2.imshow("w", img)
        cv2.waitKey(5000)
        plt.imshow(img)
        plt.show()
        img_o.save("test_result1.jpg")
if __name__ == "__main__":
    main()
