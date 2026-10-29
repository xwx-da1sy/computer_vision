"""从一张图片读取一维条码，并保存带检测框的图片和文字结果。

完整流程：
1. 确定图片路径，读取图片文件，把文件字节解码为图像数组。
2. 将原图交给 ZXing-C++，尝试寻找并解码一维条码。
3. 如果原图没有读出任何条码，分别尝试灰度、均衡化和均值滤波。
4. 从识别结果中获取条码内容、类型和位置，在原图副本上画框。
5. 为本次运行创建结果目录，保存标框图片和识别结果文本。

分工：OpenCV 负责图片读取、图像处理和绘图；ZXing-C++ 负责条码识别。
图片文件解码（JPEG/PNG 转成像素数组）和条码解码（条纹转成内容）不同。
"""
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import zxingcpp


def main():
    """返回 0 表示成功，1 表示输入或图片编码失败，2 表示未读出。"""
    # ==================== 第 1 步：确定要读取的图片 ====================
    # 脚本支持传入自己的图片路径；如果没有传入，则使用项目中的示例照片。
    # __file__ 是当前脚本路径；resolve() 返回绝对路径，parent 取上一级目录。
    script_path = Path(__file__)
    absolute_script_path = script_path.resolve()
    script_folder = absolute_script_path.parent
    default_image = script_folder / '测试图片' / 'sample_phattien.jpg'

    # ArgumentParser 读取命令行参数；不传图片路径时使用默认照片。
    # add_argument 中 nargs='?' 表示参数可省略；type=Path 将输入转换成路径对象。
    parser = argparse.ArgumentParser(description='识别图片中的一维条码')
    parser.add_argument('image', nargs='?', type=Path, default=default_image, help='图片路径；不填使用书籍照片')
    # parse_args() 根据用户在命令行输入的内容返回参数对象。
    # 例如运行 python 02_detection.py "D:\\test.jpg" 时，args.image 是该路径。
    args = parser.parse_args()
    image_path = args.image

    # is_file() 无参数；返回 True 表示该路径存在且是文件。
    if not image_path.is_file():
        print(f'图片文件不存在：{image_path}')
        return 1

    # ==================== 第 2 步：读取和解码图片文件 ====================
    # np.fromfile：输入路径字符串和字节类型，返回存放文件字节的一维数组。
    # 先读字节有助于处理当前 Windows 环境中的中文路径。
    path_string = str(image_path)
    file_data = np.fromfile(path_string, dtype=np.uint8)

    # file_data 只是 JPEG/PNG 文件的编码数据；还不能按行和列访问图片像素。
    # cv2.imdecode：输入文件字节；IMREAD_COLOR 指定按 BGR 彩色图读取。
    # 返回形状为 (高, 宽, 3) 的图像数组；无法解码时返回 None。
    # BGR 表示每个彩色像素按蓝、绿、红顺序保存；image[y, x] 可取单个像素。
    image = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
    if image is None:
        print(f'图片解码失败：{image_path}')
        return 1

    # ==================== 第 3 步：尝试识别原图 ====================
    # 一维条码用黑条与白色间隔的宽度编码数据，常见示例是 EAN-13。
    # 本程序使用现成的 ZXing-C++ 阅读器，并未自己实现条纹到数字的解码算法。
    # read_barcodes：输入图像数组；formats 指定查找的条码类型。
    # AllLinear 表示一维条码；返回成功解码的条码列表，空列表表示未读出。
    # 每个结果包含 text（内容）、format（类型）和 position（位置）。
    # 返回空列表时，不能仅凭它判断是没有定位到，还是定位后解码失败。
    barcode_formats = zxingcpp.BarcodeFormat.AllLinear
    barcodes = zxingcpp.read_barcodes(image, formats=barcode_formats)
    successful_method = '原图'
    print(f'输入图片：{image_path}')
    print(f'原图读出数量：{len(barcodes)}')

    # ==================== 第 4 步：未读出时进行预处理重试 ====================
    # if not barcodes：列表为空时条件成立；任何一步读出后就不再继续重试。
    # 如果一张图有多个条码，而原图只读出一部分，停止重试可能遗漏其他条码。
    # cv2.cvtColor：输入 BGR 图像及转换方式，返回单通道灰度图。
    # 仅当原图没有读出时尝试；灰度化不修改原图 image。
    if not barcodes:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        barcodes = zxingcpp.read_barcodes(gray_image, formats=barcode_formats)
        successful_method = '灰度图'
        print(f'灰度图读出数量：{len(barcodes)}')

    # 以下两种方法分别从灰度图出发，不叠加处理，目的是比较不同图像表示。
    # cv2.equalizeHist：输入 uint8 单通道灰度图，返回均衡化后的新图像。
    # 它根据整张图的灰度分布调整亮度；并不保证条码一定更容易识别。
    if not barcodes:
        equalized_image = cv2.equalizeHist(gray_image)
        barcodes = zxingcpp.read_barcodes(equalized_image, formats=barcode_formats)
        successful_method = '直方图均衡化'
        print(f'均衡化图读出数量：{len(barcodes)}')

    if not barcodes:
        # cv2.blur：输入灰度图及邻域大小 (3, 3)，返回均值滤波后的新图像。
        # 每个输出像素参考周围 3×3 个像素；过度平滑可能使细条纹模糊。
        blurred_image = cv2.blur(gray_image, (3, 3))
        barcodes = zxingcpp.read_barcodes(blurred_image, formats=barcode_formats)
        successful_method = '均值滤波'
        print(f'均值滤波图读出数量：{len(barcodes)}')

    # len() 返回列表中的结果个数；每个结果是一个已成功解码的条码。
    print(f'成功读取数量：{len(barcodes)}')
    if not barcodes:
        print('以上方法均未读出一维条码，请检查图片清晰度、光照和码制。')
        return 2
    print(f'成功的方法：{successful_method}')

    # ==================== 第 5 步：读取结果并画框 ====================
    # 各处理图和原图大小一致，因此返回的坐标可以直接画回原图。
    # copy() 创建图片数组的副本；画框时不修改原图。
    result_image = image.copy()
    # enumerate(..., start=1) 每次给出编号和一个结果，编号从 1 开始。
    for index, barcode in enumerate(barcodes, start=1):
        print(f'条码 {index} 类型：{barcode.format}')
        print(f'条码 {index} 内容：{barcode.text}')

        # position 包含四个角点；每个角点有 x（列）和 y（行）坐标。
        # 这与数组取像素的顺序 image[y, x] 不同。角点描述条码位置，
        # 是由 ZXing 返回的估计结果，不代表程序自己用轮廓检测定位。
        position = barcode.position
        top_left = [position.top_left.x, position.top_left.y]
        top_right = [position.top_right.x, position.top_right.y]
        bottom_right = [position.bottom_right.x, position.bottom_right.y]
        bottom_left = [position.bottom_left.x, position.bottom_left.y]
        # np.array：输入四个坐标，dtype 指定 32 位整数，返回坐标数组。
        corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

        # cv2.polylines：在 result_image 上连接 [corners] 中的角点。
        # True 表示闭合；(0, 0, 255) 是 BGR 格式的红色；3 是线宽。
        # 返回画过线的图像，同时也会直接修改传入的 result_image。
        cv2.polylines(result_image, [corners], True, (0, 0, 255), 3)

    # ==================== 第 6 步：保存图片和文字结果 ====================
    # 每次创建独立目录，避免覆盖上一次的结果。
    # strftime() 将当前时间格式化为字符串，用于生成目录名。
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_folder = script_folder / '一维条码结果' / timestamp
    # mkdir() 创建目录；parents=True 允许创建缺失的上级目录；
    # exist_ok=False 表示同名目录已存在时不覆盖，抛出异常。
    output_folder.mkdir(parents=True, exist_ok=False)
    output_path = output_folder / '检测结果.png'
    # cv2.imencode：输入文件格式和图像，返回成功标志及 PNG 字节数组。
    encoded_ok, encoded_data = cv2.imencode('.png', result_image)
    if not encoded_ok:
        print('结果图片编码失败。')
        return 1
    # tofile() 将 PNG 字节写入路径，支持这里使用的中文路径。
    output_path_string = str(output_path)
    encoded_data.tofile(output_path_string)

    # 将输入、方法、类型和内容逐行组织成文本，方便之后核对结果。
    lines = [f'输入图片：{image_path.resolve()}', f'成功的方法：{successful_method}',
             f'成功读取数量：{len(barcodes)}']
    for index, barcode in enumerate(barcodes, start=1):
        lines.append(f'条码 {index} 类型：{barcode.format}')
        lines.append(f'条码 {index} 内容：{barcode.text}')
    # join() 在每两行之间插入换行符，返回一个完整字符串。
    result_text = '\n'.join(lines) + '\n'
    text_path = output_folder / '识别结果.txt'
    text_path.write_text(result_text, encoding='utf-8')
    print(f'标框图片：{output_path}')
    print(f'文本结果：{text_path}')
    return 0


# 直接运行此文件时执行 main()；作为模块导入时不自动运行。
# SystemExit 将 main() 的返回值作为程序退出码交给操作系统。
if __name__ == '__main__':
    raise SystemExit(main())
