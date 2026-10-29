# 条码读取：新实践

当前目标是从图片中读取一维条码。`02_detection.py` 使用 OpenCV 读取图片，并使用 ZXing-C++ 定位和解码。默认示例是公开的[书籍条码照片](https://github.com/phattien12/QR-Code-Barcode-Detection-with-OpenCV/blob/main/barcode.jpg)，可读出两个 EAN-13 条码：`9784873117980` 和 `1923055034006`。少量样本的结果不能代表所有照片的识别效果。

在 `D:\computer_vision` 的 PowerShell 中运行：

```powershell
conda run --no-capture-output -n dl-study python .\任务1-5_OpenCV条形码\条码读取\02_detection.py
conda run --no-capture-output -n dl-study python .\任务1-5_OpenCV条形码\条码读取\02_detection.py "D:\你的照片.jpg"
```

程序优先读取原图；若没有读出，则依次尝试灰度、灰度图均衡化和灰度图均值滤波，每次成功即停止。多条码图片中若第一种方法只读出部分条码，程序不会继续尝试其他方法，因此可能遗漏。程序打印读出的类型和内容，并在 `一维条码结果/` 下为每次成功运行新建目录，保存标框图和识别结果文本。ZXing 返回空列表时表示没有成功读出的码，不能据此断定未找到候选区域；退出码为 2。现有 `dl-study` 已安装 OpenCV 5.0.0 和 `zxing-cpp` 3.1.1；换环境时需自行安装这两个包。

使用现有 Conda 环境 `dl-study`；预处理对照结果见 `预处理实验记录.md`。
