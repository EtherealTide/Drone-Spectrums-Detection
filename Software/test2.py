import time
import cv2
import numpy as np


def benchmark_drawing_operations():
    """测试绘制操作的耗时"""

    # 创建测试图像 (640x640x3)
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # 创建模拟检测结果（假设10个检测框）
    detections = []
    for i in range(10):
        x1, y1 = np.random.randint(0, 300, 2)
        x2, y2 = x1 + np.random.randint(50, 100), y1 + np.random.randint(50, 100)
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.9,
                "class_id": 0,
                "class_name": "drone",
            }
        )

    results = {}

    # 1. 测试图像拷贝
    t0 = time.perf_counter()
    for _ in range(100):
        annotated = image.copy()
    t_copy = (time.perf_counter() - t0) / 100 * 1000  # 毫秒
    results["copy"] = t_copy

    # 2. 测试绘制矩形框（无标签）
    t0 = time.perf_counter()
    for _ in range(100):
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    t_rect = (time.perf_counter() - t0) / 100 * 1000
    results["rect_only"] = t_rect - t_copy

    # 3. 测试绘制完整标签（包括文本）
    t0 = time.perf_counter()
    for _ in range(100):
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{det['class_name']} {det['confidence']:.2f}"
            (lw, lh), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # 标签背景
            cv2.rectangle(
                annotated, (x1, y1 - lh - baseline - 5), (x1 + lw, y1), (0, 255, 0), -1
            )

            # 文本
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
    t_full = (time.perf_counter() - t0) / 100 * 1000
    results["full_label"] = t_full - t_rect

    # 4. 测试直接修改原图（无拷贝）
    t0 = time.perf_counter()
    for _ in range(100):
        annotated = image  # 不拷贝，直接修改原图
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    t_no_copy = (time.perf_counter() - t0) / 100 * 1000
    results["no_copy"] = t_no_copy

    # 打印结果
    print("=== 绘制操作耗时分析 (毫秒) ===")
    print(f"图像拷贝: {results['copy']:.3f} ms")
    print(f"绘制矩形框: {results['rect_only']:.3f} ms")
    print(f"绘制完整标签: {results['full_label']:.3f} ms")
    print(f"总耗时(有拷贝): {t_full:.3f} ms")
    print(f"总耗时(无拷贝): {t_no_copy:.3f} ms")
    print(f"拷贝占比: {(results['copy']/t_full)*100:.1f}%")

    return results


# 运行测试
results = benchmark_drawing_operations()
