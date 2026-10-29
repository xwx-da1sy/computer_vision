"""逐步观察灰度化、均衡化、均值滤波和腐蚀对一维条码的影响。"""
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import zxingcpp

matplotlib.use('Agg')  # 将直方图保存到文件，不弹出窗口。
import matplotlib.pyplot as plt


def save_png(path, image):
    """将图片编码为 PNG 并保存，支持中文路径；失败时抛出异常。"""
    success, image_bytes = cv2.imencode('.png', image)
    if not success:
        raise ValueError(f'图片编码失败：{path}')
    image_bytes.tofile(str(path))


def main():
    script_folder = Path(__file__).resolve().parent
    default_image = script_folder / '测试图片' / 'sample_phattien.jpg'

    parser = argparse.ArgumentParser(description='观察一维条码图片的预处理')
    parser.add_argument('image', nargs='?', type=Path, default=default_image, help='图片路径；不填使用书籍照片')
    args = parser.parse_args()
    image_path = args.image
    if not image_path.is_file():
        print(f'图片文件不存在：{image_path}')
        return 1

    # 读取文件字节并解码为 BGR 彩色图；失败时 imdecode 返回 None。
    path_string = str(image_path)
    file_bytes = np.fromfile(path_string, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        print(f'图片解码失败：{image_path}')
        return 1

    # cvtColor：输入 BGR 图和转换方式，返回单通道灰度图。
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # equalizeHist：输入 uint8 单通道灰度图，返回直方图均衡化后的图。
    equalized_image = cv2.equalizeHist(gray_image)

    # blur：输入图像和邻域大小；返回 3×3 邻域均值平滑后的图。
    blurred_image = cv2.blur(equalized_image, (3, 3))

    # erode：输入图像、结构元素和迭代次数；返回腐蚀后的图。
    # 对本例灰度图，腐蚀使暗区域倾向扩大，可能破坏细条纹。
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded_image = cv2.erode(blurred_image, kernel, iterations=1)

    # calcHist：对输入图片的第 0 通道统计 256 个灰度值的像素数。
    # [256] 是箱数，[0, 256] 是取值范围；返回每个灰度值对应的频数。
    original_hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    equalized_hist = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_folder = script_folder / '预处理结果' / timestamp
    output_folder.mkdir(parents=True, exist_ok=False)

    save_png(output_folder / '01_灰度图.png', gray_image)
    save_png(output_folder / '02_均衡化.png', equalized_image)
    save_png(output_folder / '03_均值滤波.png', blurred_image)
    save_png(output_folder / '04_腐蚀.png', eroded_image)

    # 将均衡化前后的灰度直方图画在同一张图中，便于比较分布。
    plt.figure(figsize=(9, 4))
    plt.plot(original_hist, label='Before equalization')
    plt.plot(equalized_hist, label='After equalization')
    plt.xlabel('Gray value (0-255)')
    plt.ylabel('Number of pixels')
    plt.title('Grayscale histogram')
    plt.legend()
    plt.tight_layout()
    histogram_path = output_folder / '05_灰度直方图.png'
    plt.savefig(str(histogram_path), dpi=150)
    plt.close()

    # 用同一识别器比较各步骤；输出仅反映这张图片的结果。
    stages = [
        ('原图', image),
        ('灰度图', gray_image),
        ('均衡化', equalized_image),
        ('均值滤波', blurred_image),
        ('腐蚀', eroded_image),
    ]
    barcode_formats = zxingcpp.BarcodeFormat.AllLinear
    print(f'输入图片：{image_path}')
    for stage_name, stage_image in stages:
        barcodes = zxingcpp.read_barcodes(stage_image, formats=barcode_formats)
        decoded_values = [barcode.text for barcode in barcodes]
        print(f'{stage_name}识别结果：{decoded_values}')
    print(f'预处理图片和直方图保存于：{output_folder}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
