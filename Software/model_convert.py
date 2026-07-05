from ultralytics import YOLO

# 切换到软件目录
import os

os.chdir("Software")
# 加载YOLO模型并导出为OpenVINO格式
model = YOLO("best_12.14.pt")  # yolo训练好的模型为best.pt

exported = model.export(
    format="engine",
    imgsz=512,
    dynamic=True,
    batch=16,
    half=True,
    device=0,
    verbose=False,
)
# model.export(format="openvino", half=True)  # 生成 best_openvino_model/ 目录
# 提取类别名称并保存到文本文件
class_names = model.names
with open("class_names.txt", "w", encoding="utf-8") as f:
    for name in class_names.values():
        f.write(f"{name}\n")
