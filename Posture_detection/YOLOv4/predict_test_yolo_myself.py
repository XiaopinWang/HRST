import os
import json
import time
from PIL import Image
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
import datetime
from build_utils import img_utils, torch_utils, utils
from models import Darknet
from draw_box_utils import draw_box
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def main():
    img_size = 832  # 必须是32的整数倍 [416, 512, 608]
    cfg = "cfg/my_yolov4.cfg"  # 改成生成的.cfg文件
    # weights = "weights/final_yolov4spp-185.pt"  # 改成自己训练好的权重文件
    weights = "weights/final_yolov4spp-193.pt"
    # weights = "weights/YOLOv3_70_120.pt"
    json_path = "./data/pascal_voc_classes.json"  # json标签文件
    img_path = "000000000000166.jpg"
    # img_path = r"E:\pig_viedo\pig_viedo\test_viedo\img\000000000000007.jpg"
    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}
    input_size = (img_size, img_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    model.to(device)
    # results_file = "results{}_pin.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model.eval()
    with torch.no_grad():
        tensor = (torch.rand(1, 3, 832, 832),)
        # tensor = (torch.rand(1, 3, 512, 512),)
        # 分析FLOPs
        flops = FlopCountAnalysis(model, tensor)
        print("FLOPs: ", flops.total())

        # 分析parameters
        print(parameter_count_table(model))
        # init
        # img = torch.zeros((1, 3, img_size, img_size), device=device)
        # model(img)
        img_o = cv2.imread(img_path)
        assert img_o is not None, "Image Not Found " + img_path
        # # auto=True输入网路的大小并不是512*512，而是将图片的最大边长给等比例缩放到512，短边给(0, 0, 0)填充
        img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
        # # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # scale (0, 255) to (0, 1)
        img = img.unsqueeze(0)  # add batch dimension
        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]  # only get inference result
        t2 = torch_utils.time_synchronized()
        # print(t2 - t1)
        # pred = utils.non_max_suppression(pred, conf_thres=0.1, nms_thres=0.6)[0]
        pred = utils.non_max_suppression(pred, conf_thres=0.13, iou_thres=0.6, multi_label=True)[0]
        t3 = time.time()
        # print(t3 - t2)
        if pred is None:
            print("No target detected.")
            exit(0)
        # process detections
        pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
        # print(pred.shape)
        bboxes = pred[:, :4].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        classes = pred[:, 5].detach().cpu().numpy().astype(np.int32) + 1
        img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)
        plt.imshow(img_o)
        plt.show()
        # img_o.save("test_result.jpg")
        # TODO 处理视频数据
        # video_path = r"E:\pig_viedo\pig_viedo\D06_20210911060820.mp4"
        # capture = cv2.VideoCapture(video_path)
        # count_img = 0
        # while True:
        #     ref, frame = capture.read()
        # # img_o = cv2.imread(img_path)  # BGR
        #     if ref == True:
        #         img_o = frame
        #         # img_o = cv2.imread(img_path)
        #         assert img_o is not None, "Image Not Found " + img_path
        # # auto=True输入网路的大小并不是512*512，而是将图片的最大边长给等比例缩放到512，短边给(0, 0, 0)填充
        #         img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
        #         # Convert
        #         img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        #         img = np.ascontiguousarray(img)
        #         img = torch.from_numpy(img).to(device).float()
        #         img /= 255.0  # scale (0, 255) to (0, 1)
        #         img = img.unsqueeze(0)  # add batch dimension
        #         # t1 = torch_utils.time_synchronized()
        #         pred = model(img)[0]  # only get inference result
        #         # t2 = torch_utils.time_synchronized()
        #         # print(t2 - t1)
        #         # pred = utils.non_max_suppression(pred, conf_thres=0.1, nms_thres=0.6)[0]
        #         pred = utils.non_max_suppression(pred, conf_thres=0.2, iou_thres=0.6, multi_label=True)[0]
        #         t3 = time.time()
        #     # print(t3 - t2)
        #         if pred is None:
        #             print("No target detected.")
        #             exit(0)
        #         # process detections
        #         pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
        #         # print(pred.shape)
        #         bboxes = pred[:, :4].detach().cpu().numpy()
        #         scores = pred[:, 4].detach().cpu().numpy()
        #         classes = pred[:, 5].detach().cpu().numpy().astype(np.int32) + 1
        #         pig_dict = dict()
        #         inflection_list = []
        #         for index_pig in classes:
        #             if index_pig in category_index.keys():
        #                 inflection_list.append(category_index[index_pig])
        #         inflection_set = set(inflection_list)
        #         for i in inflection_set:
        #             pig_dict[i] = inflection_list.count(i)
        #         print(pig_dict, count_img)  # 添加一个序号，容易找到图片
        #         count_img += 1
        #         with open(results_file, "a") as f:
        #             txt = "{}".format(pig_dict)
        #             f.write(txt + "\n")
        #         img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)
        # #         # cv2.namedWindow('asd', cv2.WINDOW_NORMAL)
        #         cv2.namedWindow("asd", cv2.WINDOW_FREERATIO)
        #         img_o = np.asarray(img_o)
        #         img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        #         cv2.imshow("asd", img_o)
        # #         # cv2.imwrite("w1.jpg", img_o)
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
        #     else:
        #         print("切分完成!")
        #         break
        # capture.release()
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
