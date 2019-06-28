import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

np.random.seed(1000)
my_colors = []
for i in range(0, 255, 15):
    for j in range(0, 255, 15):
        for k in range(0, 255, 15):
            my_colors.append((i, j, k))
my_colors = [random.choice(my_colors) for _ in range(1000)]


def get_dets(num_bb=100):
    '''
    :param num_bb:  number of bounding  box
    :return: boxes: [(x1, y1, x2, y2, score, idx), ...]  (score: random)
    '''

    dets = np.zeros((num_bb, 6), dtype=np.float)

    X1 = np.random.randint(0, 500, size=20)
    Y1 = np.random.randint(0, 500, size=20)
    X2 = np.random.randint(500, 1000, size=20)
    Y2 = np.random.randint(500, 1000, size=20)
    scores = np.random.rand(20)

    for i, box in enumerate(zip(X1, Y1, X2, Y2, scores)):
        dets[i, 0], dets[i, 1] = box[0], box[1]  # x1 y1
        dets[i, 2], dets[i, 3] = box[2], box[3]  # x2 y2
        dets[i, 4], dets[i, 5] = box[4], i  # s  idx

    return dets

def plot_rect(rects, result_name):
    '''
        :param rects:        shape  [(x1, y1, x2, y2, ...), ...]
        :param result_name:  saving the result image
        :return:             None
    '''
    plt.figure(figsize=(8, 8))
    canvas = 100 * np.ones((1000, 1000, 3))

    for i, rect1 in enumerate(rects):
        x1, y1 = int(rect1[0]), int(rect1[1])
        x2, y2 = int(rect1[2]), int(rect1[3])
        font = cv2.FONT_HERSHEY_SIMPLEX
        _ = cv2.rectangle(canvas, (x1, y1), (x2, y2), my_colors[i], 2)
        # img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin
        _ = cv2.putText(img=canvas,
                        text=str(int(rect1[5])),
                        org=(x1, y1 + 7),
                        fontFace=font,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=2)

        cv2.imwrite('./' + result_name, canvas)
        plt.imshow(canvas)

    plt.show()

def faster_nms(dets, threshold=0.5):
    '''
        NMS算法简要：
            1. 按照预测的置信度(是否有物体的得分)进行排序，得到对应的指针
            2. 遍历第一个边界框的指针，保留第一个指针，求与剩余边界框的IoU，根据IoU去掉大于阈值多个边界框的指针
            3. 再遍历第二个指针，并重复上面的操作
    '''

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 获取从大到小排序的指针
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], y2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # keep the bounding boxes that smalling than thresold
        inds = np.where(ovr < threshold)[0]
        order = order[inds + 1]

    return keep

def test():
    dets = get_dets()
    plot_rect(dets, 'before.png')
    keep = faster_nms(dets)
    print('keeping number:', len(keep))
    print(keep)
    new_dets = dets[keep]
    plot_rect(new_dets, 'after.png')


if __name__ == '__main__':
    test()