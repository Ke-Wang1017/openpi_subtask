#!/usr/bin/env python3
"""
创建测试图像文件
"""

import os

import cv2
import numpy as np


def create_test_images():
    """创建测试用的图像文件"""
    script_dir = "/root/workspace/chenyj36@xiaopeng.com/openpi_subtask/scripts"

    # 创建不同颜色的测试图像
    images = {
        "faceImg.png": (255, 0, 0),  # 红色
        "leftImg.png": (0, 255, 0),  # 绿色
        "rightImg.png": (0, 0, 255),  # 蓝色
    }

    for filename, color in images.items():
        # 创建224x224的彩色图像
        img = np.full((224, 224, 3), color, dtype=np.uint8)

        # 添加一些纹理
        noise = np.random.randint(-30, 30, (224, 224, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 添加一些几何形状
        if "face" in filename:
            # 添加圆形(模拟人脸)
            cv2.circle(img, (112, 112), 50, (255, 255, 255), -1)
            cv2.circle(img, (100, 100), 5, (0, 0, 0), -1)  # 左眼
            cv2.circle(img, (124, 100), 5, (0, 0, 0), -1)  # 右眼
            cv2.ellipse(img, (112, 120), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # 嘴巴
        elif "left" in filename:
            # 添加矩形
            cv2.rectangle(img, (50, 50), (174, 174), (255, 255, 255), 3)
        elif "right" in filename:
            # 添加三角形
            pts = np.array([[112, 50], [50, 174], [174, 174]], np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=3)

        # 保存图像
        filepath = os.path.join(script_dir, filename)
        cv2.imwrite(filepath, img)
        print(f"✅ 创建测试图像: {filepath}")


if __name__ == "__main__":
    create_test_images()
    print("🎉 所有测试图像创建完成!")
