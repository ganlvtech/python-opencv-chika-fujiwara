import cv2
import numpy as np


def hist_average(array, bins):
    hist, bins = np.histogram(array, bins)
    i = np.argmax(hist)
    ret = np.average(list(filter(lambda x: bins[i] <= x <= bins[i + 1], array)))
    return ret


def match_bg_key_point(bg_kp, frame_kp, matches_sorted):
    if len(matches_sorted) < 2:
        return None, None, None
    scales = np.zeros(int(len(matches_sorted) * (len(matches_sorted) - 1) / 2))
    k = 0
    for i in range(0, len(matches_sorted) - 1):
        for j in range(i + 1, len(matches_sorted)):
            # 所有特征点两两对比，计算距离，当前帧特征点距离与背景特征点距离的比，就是图片缩放的比例
            frame_kp1 = frame_kp[matches_sorted[i].queryIdx].pt
            frame_kp2 = frame_kp[matches_sorted[j].queryIdx].pt
            bg_kp1 = bg_kp[matches_sorted[i].trainIdx].pt
            bg_kp2 = bg_kp[matches_sorted[j].trainIdx].pt
            frame_ds = np.hypot(frame_kp1[0] - frame_kp2[0], frame_kp1[1] - frame_kp2[1])
            bg_ds = np.hypot(bg_kp1[0] - bg_kp2[0], bg_kp1[1] - bg_kp2[1])
            # 两点距离必须超过 100，否则这两点太近，误差太大
            if frame_ds >= 100 and bg_ds >= 100:
                scales[k] = frame_ds / bg_ds
                k = k + 1
    scales = scales[:k]
    # 挑选直方图中分布点最多的区间取平均值
    frame_bg_scale = hist_average(scales, 7)

    frame_bg_dxs = np.zeros(len(matches_sorted))
    frame_bg_dys = np.zeros(len(matches_sorted))
    for i in range(0, len(matches_sorted)):
        frame_kp1 = frame_kp[matches_sorted[i].queryIdx].pt
        bg_kp1 = bg_kp[matches_sorted[i].trainIdx].pt
        # 将当前帧每一个特征点按比例缩放到和背景一样大，然后计算与背景特征点的距离
        frame_bg_dxs[i] = bg_kp1[0] - frame_kp1[0] / frame_bg_scale
        frame_bg_dys[i] = bg_kp1[1] - frame_kp1[1] / frame_bg_scale
    # 挑选直方图中分布点最多的区间取平均值
    frame_bg_dx = hist_average(frame_bg_dxs, 5)
    frame_bg_dy = hist_average(frame_bg_dys, 5)
    if frame_bg_scale is None or frame_bg_dx is None or frame_bg_dy is None:
        return None, None, None
    return frame_bg_scale, frame_bg_dx, frame_bg_dy


def calc_bg_view_box(frame_w, frame_h, frame_bg_scale, frame_bg_dx, frame_bg_dy):
    view_box_left = frame_bg_dx
    view_box_top = frame_bg_dy
    view_box_right = frame_bg_dx + frame_w / frame_bg_scale
    view_box_bottom = frame_bg_dy + frame_h / frame_bg_scale
    return int(view_box_left), int(view_box_top), int(view_box_right), int(view_box_bottom)


# 读取背景图片
bg = cv2.imread('bg.jpg')
bg_h, bg_w, _ = bg.shape
# 打开视频
cap = cv2.VideoCapture('av42322248.mp4')
# cap.set(cv2.CAP_PROP_POS_FRAMES, 177)
# 打开输出视频文件。一次请只使用一个文件，仅写入一个文件。如果同时使用多个文件，同时写入多个文件的话，一旦 try-except 触发，可能会多出 1 帧。
# out_matches_img = cv2.VideoWriter('chika_matches_img.mp4', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), cap.get(cv2.CAP_PROP_FPS), (1616, 385))
out_only_bg = cv2.VideoWriter('chika_only_bg.mp4', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), cap.get(cv2.CAP_PROP_FPS), (1920, 814))
# out_full_bg = cv2.VideoWriter('chika_full_bg.mp4', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), cap.get(cv2.CAP_PROP_FPS), (bg_w, bg_h))

# 创建 ORB 特征点识别器
orb = cv2.ORB_create()
# 识别背景图片特征点
bg_kp, bg_des = orb.detectAndCompute(bg, None)
# 创建暴力匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 与前后帧对比，用于稳定画面
last_2_frame_bg_scale = None
last_2_frame_bg_dx = None
last_2_frame_bg_dy = None
last_frame_bg_scale = None
last_frame_bg_dx = None
last_frame_bg_dy = None
this_frame_bg_scale = None
this_frame_bg_dx = None
this_frame_bg_dy = None
# 初始化
last_frame = None
last_frame_kp = None
last_matches_sorted = None
frame = None
frame_kp = None
matches_sorted = None
cap_finished_count = 0

while cv2.waitKey(1) != ord('q'):
    try:
        last_2_frame_bg_scale = last_frame_bg_scale
        last_2_frame_bg_dx = last_frame_bg_dx
        last_2_frame_bg_dy = last_frame_bg_dy
        last_frame_bg_scale = this_frame_bg_scale
        last_frame_bg_dx = this_frame_bg_dx
        last_frame_bg_dy = this_frame_bg_dy
        last_frame = frame
        last_frame_kp = frame_kp
        last_matches_sorted = matches_sorted
        # 读取每一帧
        ret, frame = cap.read()
        if not ret:
            # 因为有稳定器，所以要多计算 1 帧，不能立刻停止
            cap_finished_count = cap_finished_count + 1
            if cap_finished_count >= 2:
                break
            this_frame_bg_scale = None
            this_frame_bg_dx = None
            this_frame_bg_dy = None
        else:
            # 当前帧序号
            print(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 去除黑边，视频格式为 1920 x 1080，处理之后为 1920 x 814
            frame = frame[133:-133, :, :]
            frame_h, frame_w, _ = frame.shape
            cv2.imshow('frame', cv2.resize(frame, (0, 0), fx=0.4, fy=0.4))

            # 识别每一帧的特征点
            frame_kp, frame_des = orb.detectAndCompute(frame, None)

            # 将帧特征点与背景特征点比较，按相似程度排列，只取比较合理的点
            matches = bf.match(frame_des, bg_des)
            matches_sorted = sorted(matches, key=lambda x: x.distance)
            matches_sorted = list(filter(lambda x: x.distance < 20, matches_sorted))

            # 计算相同区域缩放和偏移
            this_frame_bg_scale, this_frame_bg_dx, this_frame_bg_dy = match_bg_key_point(bg_kp, frame_kp, matches_sorted)

        if last_frame is None:
            continue

        # ================================================================================
        # 以下均计算的内容均为前 1 帧，生成的画面也是前一帧的画面

        # 通过当前帧和前 2 帧来稳定前 1 帧
        if last_frame_bg_scale is None:
            last_frame_bg_scale = this_frame_bg_scale
            last_frame_bg_dx = this_frame_bg_dx
            last_frame_bg_dy = this_frame_bg_dy
            if last_2_frame_bg_scale is not None and this_frame_bg_scale is not None:
                frame_bg_scale = np.average((last_2_frame_bg_scale, this_frame_bg_scale))
                frame_bg_dx = int(np.average((last_2_frame_bg_dx, this_frame_bg_dx)))
                frame_bg_dy = int(np.average((last_2_frame_bg_dy, this_frame_bg_dy)))
            else:
                frame_bg_scale = None
                frame_bg_dx = None
                frame_bg_dy = None
        else:
            if last_2_frame_bg_scale is None or this_frame_bg_scale is None:
                frame_bg_scale = last_frame_bg_scale
                frame_bg_dx = last_frame_bg_dx
                frame_bg_dy = last_frame_bg_dy
            else:
                if last_2_frame_bg_scale < last_frame_bg_scale < this_frame_bg_scale or last_2_frame_bg_scale > last_frame_bg_scale > this_frame_bg_scale:
                    frame_bg_scale = last_frame_bg_scale
                else:
                    frame_bg_scale = np.average((last_2_frame_bg_scale, this_frame_bg_scale, last_frame_bg_scale))
                if last_2_frame_bg_dx < last_frame_bg_dx < this_frame_bg_dx or last_2_frame_bg_dx > last_frame_bg_dx > this_frame_bg_dx:
                    frame_bg_dx = last_frame_bg_dx
                else:
                    frame_bg_dx = int(np.average((last_2_frame_bg_dx, this_frame_bg_dx, last_frame_bg_dx)))
                if last_2_frame_bg_dy < last_frame_bg_dy < this_frame_bg_dy or last_2_frame_bg_dy > last_frame_bg_dy > this_frame_bg_dy:
                    frame_bg_dy = last_frame_bg_dy
                else:
                    frame_bg_dy = int(np.average((last_2_frame_bg_dy, this_frame_bg_dy, last_frame_bg_dy)))

        # 计算相同区域
        if frame_bg_scale is None:
            # out_matches_img.write(cv2.resize(last_frame, (1616, 385)))
            out_only_bg.write(last_frame)
            # out_full_bg.write(bg)
        else:
            view_box_left, view_box_top, view_box_right, view_box_bottom = calc_bg_view_box(frame_w, frame_h, frame_bg_scale, frame_bg_dx, frame_bg_dy)

            # # 绘制方框
            # bg_framed = cv2.copyTo(bg, None)
            # cv2.rectangle(bg_framed, (view_box_left, view_box_top), (view_box_right, view_box_bottom), (0, 255, 0), 2)
            # # 绘制匹配结果
            # matches_img = cv2.drawMatches(last_frame, last_frame_kp, bg_framed, bg_kp, last_matches_sorted, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # matches_img = cv2.resize(matches_img, (1616, 385))
            # cv2.imshow('matches_img', matches_img)
            # out_matches_img.write(matches_img)

            # 提取相同区域
            same_region = bg[view_box_top:view_box_bottom, view_box_left:view_box_right, :]
            same_region = cv2.resize(same_region, (frame_w, frame_h))
            cv2.imshow('same_region', cv2.resize(same_region, (0, 0), fx=0.4, fy=0.4))
            out_only_bg.write(same_region)

            # # 把当前帧合成到整个背景
            # frame_resize_to_bg = cv2.resize(frame, (view_box_right - view_box_left, view_box_bottom - view_box_top))
            # frame_with_full_bg = cv2.copyTo(bg, None)
            # frame_with_full_bg[view_box_top:view_box_bottom, view_box_left:view_box_right] = frame_resize_to_bg
            # cv2.imshow('frame_with_full_bg', cv2.resize(frame_with_full_bg, (0, 0), fx=0.4, fy=0.4))
            # out_full_bg.write(frame_with_full_bg)

    except Exception as e:
        print(e)
        cv2.waitKey(0)
