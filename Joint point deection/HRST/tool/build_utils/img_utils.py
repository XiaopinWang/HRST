import numpy as np
import cv2


def letterbox(img: np.ndarray,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放调整到指定大小
    :param img: 输入的图像numpy格式
    :param new_shape: 输入网络的shape
    :param color: padding用什么颜色填充
    :param auto:
    :param scale_fill: 简单粗暴缩放到指定大小
    :param scale_up:  只缩小，不放大
    :return:
    """

    shape = img.shape[:2]  # [h, w] (1440, 2560)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # (0.5777777777777777, 0.325)
    if not scale_up:  # only scale down, do not scale up (for better test mAP) 对于大于指定输入大小的图片进行缩放,小于的不变
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios  # (0.325, 0.325)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (832, 468)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]   # (0, 364) # wh padding
    if auto:  # minimun rectangle 保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍(416x416)，如果是(512x512)可以保证是64的整数倍
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)   # (0, 44) # wh padding
    elif scale_fill:  # stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # wh ratios

    dw /= 2  # divide padding into 2 sides 将padding分到上下，左右两侧
    dh /= 2  # 22

    # shape:[h, w]  new_unpad:[w, h]
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # 双线性插值（默认设置） # (468, 832, 3)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding 22
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding 0

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # (512, 832, 3)  # add border
    """
    src ： 输入的图片
    top, bottom, left, right ：相应方向上的边框宽度
    borderType：定义要添加边框的类型，它可以是以下的一种：
    cv2.BORDER_CONSTANT：添加的边界框像素值为常数（需要额外再给定一个参数）
    cv2.BORDER_REFLECT：添加的边框像素将是边界元素的镜面反射，类似于gfedcb|abcdefgh|gfedcba
    cv2.BORDER_REFLECT_101 or cv2.BORDER_DEFAULT：和上面类似，但是有一些细微的不同，类似于gfedcb|abcdefgh|gfedcba
    cv2.BORDER_REPLICATE：使用最边界的像素值代替，类似于aaaaaa|abcdefgh|hhhhhhh
    cv2.BORDER_WRAP：不知道怎么解释，直接看吧，cdefgh|abcdefgh|abcdefg
    value：如果borderType为cv2.BORDER_CONSTANT时需要填充的常数值
    """
    return img, ratio, (dw, dh)








