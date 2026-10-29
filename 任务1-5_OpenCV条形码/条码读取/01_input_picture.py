from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 以脚本所在目录为起点，找到测试照片。
script_path = Path(__file__)
absolute_script_path = script_path.resolve()
script_folder = absolute_script_path.parent
image_path = script_folder / "测试图片" / "sample_phattien.jpg"

# is_file()：无参数；返回 True 表示路径指向一个存在的文件。
if not image_path.is_file():
    raise FileNotFoundError(f"图片文件不存在：{image_path}")

# str(image_path)：把 Path 对象转换为路径字符串。
path_string = str(image_path)

# np.fromfile()：读取文件里的原始字节。
# 第一个参数是文件路径，dtype=np.uint8 表示按无符号字节读取。
# 返回一维 NumPy 数组；此时是 PNG 文件数据，还不是图片像素数组。
file_data = np.fromfile(path_string, dtype=np.uint8)

# cv2.imdecode()：将图片文件数据解码为图像数组。
# 第一个参数是文件数据，第二个参数指定按彩色图片读取。
# 成功返回形状为 (高, 宽, 3) 的 NumPy 数组；解码失败返回 None。
image = cv2.imdecode(file_data, cv2.IMREAD_COLOR)

if image is None:
    raise ValueError(f"文件存在，但图片解码失败：{image_path}")

print("图片路径：", image_path)
print("读取成功")
print("图片形状（高、宽、通道数）：", image.shape)
print("像素数据类型：", image.dtype)

# cv2.cvtColor()：转换图像的颜色表示。
# 第一个参数是输入图像；第二个参数指定从 BGR 彩色图转为灰度图。
# 返回一个新的单通道 NumPy 数组；原图 image 不会被修改。
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("灰度图形状（高、宽）：", gray_image.shape)
print("灰度图像素数据类型：", gray_image.dtype)

# 用灰度色图显示灰度数组；vmin 和 vmax 指定亮度范围。
plt.imshow(gray_image, cmap="gray", vmin=0, vmax=255)
plt.title("Grayscale image")
plt.show()
