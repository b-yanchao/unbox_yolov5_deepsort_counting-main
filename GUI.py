import os
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import Combobox

from PIL import Image, ImageTk
import matplotlib.pyplot as plt


def center_window(top, width, height):
    screenwidth = top.winfo_screenwidth()
    screenheight = top.winfo_screenheight()
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    top.geometry(size)

top = Tk()
top.title('车辆计数系统')


center_window(top, 700, 580)
top.maxsize(700, 580)
top.minsize(700, 580)
videosign = False
camerasign = False
video_name = "请选择视频文件"

#预测视频函数
def predictvideo():

    # 可用来存储和处理大型矩阵
    import numpy as np
    # tqdm模块是python进度条库，添加加载视频的进度条
    from tqdm import tqdm
    # deepsort跟踪和efficientnet二次分类
    import tracker
    # yolov5第识别车辆
    from detector import Detector
    import cv2
    import os
    global capture
    global videoWriter

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 根据视频尺寸，填充一个polygon，供撞线计算使用,np.zeros返回来一个给定形状和类型的用0填充的数组；
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)   #测试集视频分辨率
    #list_pts_blue = [[780, 0], [860, 0], [860, 1080], [780, 1080]]  # 撞线的四个坐标点
    list_pts_blue = [[820, 0], [940, 0], [940, 1080], [820, 1080]]  # 撞线的四个坐标点
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)  # 根据四个坐标点初始化数据
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)  # 在mask_image_temp中填充出多边形（ndarray_pts_blue），color为撞线后记录的编号，（1080,1920
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]  # 给原数组增加维度,将2维矩阵中的每一个元素拿出来，单独放一个数据里，将原来二维的转化为3维的,方便后面撞线时根据检测框的坐标值来判断是否撞线，（1080,1920,1）


    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    # 初始化撞线polygon
    list_pts_yellow = [[940, 0], [1060, 0], [1060, 1080], [940, 1080]]  # 撞线的四个坐标点
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)  # 根据四个坐标点初始化数据
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)  # 在mask_image_temp中填充出多边形（ndarray_pts_blue），color为撞线后记录的编号
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]  # 给原数组增加维度
    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 黄 色盘
    yellow_color_plate = [0, 255, 255]

    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image

    # list 与蓝色polygon重叠  记录撞线的汽车编号
    list_overlapping_blue_polygon = []
    list_overlapping_yellow_polygon = []

    count = 0
    cls_count = [0, 0, 0, 0, 0, 0]  # 记录分类后每种类型车辆的数量
    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX  # 字体
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))  # 位置

    # 初始化 yolov5
    detector = Detector()
    #将标签值置零
    labeltext.config(text="Count: 0|xkc: 0|dkc: 0|xhc: 0|dhc: 0|jzxc: 0|zhc: 0")
    labeltext.config(font=("宋体", 12))

    #若未选择视频，提示错误，并退出函数
    if(video_name == "请选择视频文件"):
        messagebox.showerror('错误', '未选择视频文件')
        return

    capture = cv2.VideoCapture("video/"+video_name)
    cler = 0
    videoWriter = None
    t1 = tqdm(total=60 * 12 * 25)  # 加载进度条

    #创建展示预测结果的Label控件
    global lmain,videosign
    videosign = True
    lmain = Label(canvas, width=680, height=450)
    lmain.pack()

    while True:
        t1.update(1)
        cler += 1
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break

        # 规定尺寸，1920x1080
        im = cv2.resize(im, (1920, 1080))

        list_bboxs = []
        bboxes = detector.detect(im)  # 调用yolov5识别每帧图片中的汽车

        # 如果画面中 有bbox
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)  # 调用deepsort重识别车辆，并返回新的坐标、类别和编号
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)  # 追踪更新标注框，并标注出类别和编号
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        if cler >= 250:
            list_overlapping_blue_polygon.clear()
            cler = 0

        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)
        # 分类标签
        #dict_cls = {0: 'xkc', 1: 'dkc', 2: 'xhc', 3: 'dhc', 4: 'jzxc', 5: 'zhc'}
        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, cls_id, track_id = item_bbox
                if polygon_mask_blue_and_yellow[y2, x2] == 1 and track_id not in list_overlapping_yellow_polygon:  # 撞蓝线且未撞过黄线
                    # 如果撞 蓝polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                        list_overlapping_yellow_polygon.append(track_id)
                        # if track_id in list_overlapping_yellow_polygon:
                        count += 1
                        cls_count[int(cls_id)] += 1
                elif polygon_mask_blue_and_yellow[y2, x2] == 2 and track_id not in list_overlapping_blue_polygon:  # 撞黄线且未撞过蓝线
                    # 如果撞 黄polygon
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                        list_overlapping_yellow_polygon.append(track_id)
                        # if track_id in list_overlapping_blue_polygon:
                        count += 1
                        cls_count[int(cls_id)] += 1

        # 更新计数
        text_draw = 'Count: ' + str(count) + '|' + 'xkc: ' + str(cls_count[0]) + '|' + 'dkc: ' + str(
            cls_count[1]) + '|' + 'xhc: ' + str(cls_count[2]) + '|' + 'dhc: ' + str(
            cls_count[3]) + '|' + 'jzxc: ' + str(cls_count[4]) + '|zhc: ' + str(cls_count[5])
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(255, 255, 0), thickness=2)

        #显示计数数量
        labeltext.config(text=text_draw)
        labeltext.config(font=("宋体", 12))


        # 将识别结果生成mp4视频
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result_video.mp4', fourcc, 25, (output_image_frame.shape[1], output_image_frame.shape[0]))
        # 显示每帧识别结果
        videoWriter.write(output_image_frame)
        # cv2.imshow('车辆计数', cv2.resize(output_image_frame, (960, 540)))

        #显示每帧预测结果，逐帧替换
        img = cv2.cvtColor(output_image_frame, cv2.COLOR_BGR2RGBA)  # 转换颜色使播放时保持原有色彩
        current_image = Image.fromarray(img).resize((680, 450))  # 将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        lmain.imgtk = imgtk
        lmain.config(image=imgtk)
        lmain.update()

    capture.release()
    videoWriter.release()

#打开摄像头预测
def predictcamera():
    # 可用来存储和处理大型矩阵
    import numpy as np
    # tqdm模块是python进度条库，添加加载视频的进度条
    from tqdm import tqdm
    # deepsort跟踪和efficientnet二次分类
    import tracker
    # yolov5第识别车辆
    from detector import Detector
    import cv2
    import os
    global capturec
    global videoWriterc

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # 根据视频尺寸，填充一个polygon，供撞线计算使用,np.zeros返回来一个给定形状和类型的用0填充的数组；
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    # list_pts_blue = [[780, 0], [860, 0], [860, 1080], [780, 1080]]  # 撞线的四个坐标点
    list_pts_blue = [[820, 0], [940, 0], [940, 1080], [820, 1080]]  # 撞线的四个坐标点
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)  # 根据四个坐标点初始化数据
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)  # 在mask_image_temp中填充出多边形（ndarray_pts_blue），color为撞线后记录的编号
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]  # 给原数组增加维度

    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    # 初始化撞线polygon
    list_pts_yellow = [[940, 0], [1060, 0], [1060, 1080], [940, 1080]]  # 撞线的四个坐标点
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)  # 根据四个坐标点初始化数据
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)  # 在mask_image_temp中填充出多边形（ndarray_pts_blue），color为撞线后记录的编号
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]  # 给原数组增加维度
    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 黄 色盘
    yellow_color_plate = [0, 255, 255]

    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image

    # list 与蓝色polygon重叠  记录撞线的汽车ID
    list_overlapping_blue_polygon = []
    list_overlapping_yellow_polygon = []

    # 进入数量
    down_count = 0
    # 离开数量
    up_count = 0
    # 总数量
    count = 0
    cls_count = [0, 0, 0, 0, 0, 0]  # 记录二次分类后每种类型车辆的数量
    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX  # 字体
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))  # 位置

    # 初始化 yolov5
    detector = Detector()
    # 将标签值置零
    labeltext.config(text="Count: 0|xkc: 0|dkc: 0|xhc: 0|dhc: 0|jzxc: 0|zhc: 0")
    labeltext.config(font=("宋体", 12))

    capturec = cv2.VideoCapture(0)
    cler = 0
    videoWriterc = None
    t1 = tqdm(total=60 * 12 * 25)  # 加载进度条
    global lmainc,camerasign
    camerasign = True
    lmainc = Label(canvas, width=680, height=450)
    lmainc.pack()
    while True:
        t1.update(1)
        cler += 1
        # 读取每帧图片
        _, im = capturec.read()
        if im is None:
            break

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (1920, 1080))

        list_bboxs = []
        bboxes = detector.detect(im)  # 调用yolov5识别每帧图片中的汽车

        # 如果画面中 有bbox
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)  # 调用deepsort重识别车辆，并返回新的坐标、类别和编号

            # 画框、分类并将分类结果标注到框上
            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)  # 追踪更新标注框，并标注出类别和编号
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        if cler >= 250:
            list_overlapping_blue_polygon.clear()
            cler = 0

        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)
        # 二次分类标签
        #dict_cls = {0: 'xkc', 1: 'dkc', 2: 'xhc', 3: 'dhc', 4: 'jzxc', 5: 'zhc'}
        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, cls_id, track_id = item_bbox
                if polygon_mask_blue_and_yellow[y2, x2] == 1 and track_id not in list_overlapping_yellow_polygon:  # 撞蓝线且未撞过黄线
                    # 如果撞 蓝polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                        list_overlapping_yellow_polygon.append(track_id)
                        # if track_id in list_overlapping_yellow_polygon:
                        count += 1
                        cls_count[int(cls_id)] += 1
                elif polygon_mask_blue_and_yellow[y2, x2] == 2 and track_id not in list_overlapping_blue_polygon:  # 撞黄线且未撞过蓝线
                    # 如果撞 黄polygon
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                        list_overlapping_yellow_polygon.append(track_id)
                        # if track_id in list_overlapping_blue_polygon:
                        count += 1
                        cls_count[int(cls_id)] += 1
        # 更新计数
        text_draw = 'Count: ' + str(count) + '|' + 'xkc: ' + str(cls_count[0]) + '|' + 'dkc: ' + str(
            cls_count[1]) + '|' + 'xhc: ' + str(cls_count[2]) + '|' + 'dhc: ' + str(
            cls_count[3]) + '|' + 'jzxc: ' + str(cls_count[4]) + '|zhc: ' + str(cls_count[5])
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(255, 255, 255), thickness=2)

        labeltext.config(text=text_draw)
        labeltext.config(font=("宋体", 12))

        # 将识别结果生成mp4视频
        if videoWriterc is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriterc = cv2.VideoWriter(
                'result_camera.mp4', fourcc, 25, (output_image_frame.shape[1], output_image_frame.shape[0]))
        # 显示每帧识别结果
        videoWriterc.write(output_image_frame)
        # cv2.imshow('车辆计数', cv2.resize(output_image_frame, (960, 540)))
        img = cv2.cvtColor(output_image_frame, cv2.COLOR_BGR2RGBA)  # 转换颜色使播放时保持原有色彩
        current_image = Image.fromarray(img).resize((680, 450))  # 将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        lmainc.imgtk = imgtk
        lmainc.config(image=imgtk)
        lmainc.update()

    capturec.release()
    videoWriterc.release()
#点击关闭视频
def clossvideo():
    global videosign
    lmain.destroy()
    capture.release()
    videoWriter.release()
    videosign = False
#点击关闭摄像头
def closscamera():
    global camerasign
    lmainc.destroy()
    capturec.release()
    videoWriterc.release()
    camerasign = False
#点击退出键
def exit():
    if(videosign == True):
       clossvideo()
    if(camerasign == True):
        closscamera()
    top.quit()
#读取测试集视频名字
def file_name(file_dir):
    li = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.mp4':
                li.append(os.path.join(file))
    return li
#下拉列表触发事件
def xFunc(event):
    global video_name
    video_name = xVariable.get()


#基本页面布局
canvas = Canvas(top, bg='white', width=680, height=450)
canvas.place(x=8,y=25)

labeltext = Label(top,text="Count: 0|xkc: 0|dkc: 0|xhc: 0|dhc: 0|jzxc: 0|zhc: 0")
labeltext.config(font=("宋体", 12))
labeltext.place(x=8,y=5)

xVariable = StringVar()  # #创建变量，便于取值

com = Combobox(top, textvariable=xVariable)
com.place(x=50,y=490)
li = file_name("video/")
filename = ["请选择视频文件"]
for i in range(len(li)):
    filename.append(li[i])
com["value"] = filename
com.current(0)


com.bind("<<ComboboxSelected>>", xFunc)  # #给下拉菜单绑定事件

button2 = Button(top, text="识别视频", activeforeground='red', relief=RIDGE, command=predictvideo)
button2.place(x=300, y=490)
button3 = Button(top, text="关闭视频", activeforeground='red', relief=RIDGE, command=clossvideo)
button3.place(x=450, y=490)
button2 = Button(top, text="打开摄像头", activeforeground='red', relief=RIDGE, command=predictcamera)
button2.place(x=295, y=540)
button3 = Button(top, text="关闭摄像头", activeforeground='red', relief=RIDGE, command=closscamera)
button3.place(x=445, y=540)
button4 = Button(top, text="退出", activeforeground='red', relief=RIDGE, command=exit, height=4, width=10)
button4.place(x=600, y=490)
top.mainloop()