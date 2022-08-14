#可用来存储和处理大型矩阵
import numpy as np
#tqdm模块是python进度条库，添加加载视频的进度条
from tqdm import tqdm
#deepsort跟踪和efficientnet二次分类
import tracker
#yolov5第一次分类
from detector import Detector
import cv2
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if __name__ == '__main__':

    # 根据视频尺寸，填充一个polygon，供撞线计算使用,np.zeros返回来一个给定形状和类型的用0填充的数组；
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    # 初始化撞线polygon
    list_pts_blue = [[800, 0], [960, 0], [960, 1080], [800, 1080]]#撞线的四个坐标点
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)#根据四个坐标点初始化数据
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)#在mask_image_temp中填充出多边形（ndarray_pts_blue），color为撞线后记录的编号
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]#给原数组增加维度
    #print(polygon_blue_value_1)


    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1

    # 缩小尺寸，1920x1080->960x540
    # polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 黄 色盘
    yellow_color_plate = [0, 255, 255]

    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)
    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image
    # 缩小尺寸，1920x1080->960x540
    # color_polygons_image = cv2.resize(color_polygons_image, (384, 640))

    # list 与蓝色polygon重叠  记录撞线的汽车ID
    list_overlapping_blue_polygon = []
    list_overlapping_yellow_polygon = []

    #总数量
    count = 0
    cls_count = [0, 0, 0, 0, 0, 0]#记录二次分类后每种类型车辆的数量
    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX#字体
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))#位置

    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture('video/433267.mp4')
    cler = 0
    videoWriter = None
    t1 = tqdm(total=60 * 12 * 25)#加载进度条
    while True:
        t1.update(1)
        cler += 1
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (1920, 1080))

        list_bboxs = []
        bboxes = detector.detect(im)#调用yolov5识别每帧图片中的汽车

        # 如果画面中 有bbox
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)#调用deepsort重识别车辆，并返回新的坐标、类别和编号
            # 画框、分类并将分类结果标注到框上
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)#追踪更新标注框，并标注出类别和编号
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        if cler >= 250:
            list_overlapping_blue_polygon.clear()
            cler = 0

        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)
        #二次分类标签
        dict_cls = {0: 'xkc', 1: 'dkc', 2: 'xhc', 3: 'dhc', 4: 'jzxc', 5: 'zhc'}
        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, cls_id, track_id = item_bbox
                #y_offset = int(y1 + ((y2 - y1) * 0.5))  # 设置一个撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                # 撞线的点
                if polygon_mask_blue_and_yellow[y2, x1] == 1 or polygon_mask_blue_and_yellow[y2, x2] == 1:  # 撞线
                    # 如果撞 蓝polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                        count += 1
                        cls_count[int(cls_id)] += 1

        #更新计数
        text_draw = 'Count: ' + str(count) + '|' + 'xkc: ' + str(cls_count[0]) + '|' + 'dkc: ' + str(
            cls_count[1]) + '|' + 'xhc: ' + str(cls_count[2]) + '|' + 'dhc: ' + str(
            cls_count[3]) + '|' + 'jzxc: ' + str(cls_count[4])+ '|zhc: ' + str(cls_count[5])
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(255, 255, 255), thickness=2)
        '''
                if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, 25, (output_image_frame.shape[1], output_image_frame.shape[0]))
        #显示每帧识别结果
        videoWriter.write(output_image_frame)
        
        
        '''
        #将识别结果生成mp4视频
        cv2.imshow('车辆计数', cv2.resize(output_image_frame, (960, 540)))
        cv2.waitKey(1)

    capture.release()
    videoWriter.release()
    cv2.destroyAllWindows()
