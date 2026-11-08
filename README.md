# Android 端部署说明（IMU-HAR / TFLite int8）

本项目在 `android/` 下提供完整的 Android Studio 工程（Java），实现 IMU 人体活动识别的**端侧 int8 量化推理**与**推理测试**。

## 1. 目录与文件

```
android/
├── settings.gradle / build.gradle         # 工程与仓库配置
└── app/
    ├── build.gradle                       # 含 TensorFlow Lite 依赖 + aaptOptions.noCompress
    └── src/main/
        ├── AndroidManifest.xml
        ├── java/com/example/imuhar/
        │   ├── HARClassifier.java         # 模型加载 + 标准化 + int8 推理封装
        │   └── MainActivity.java          # 推理测试界面
        ├── res/                           # 布局 / 主题 / 字符串
        └── assets/
            ├── har_cnn_int8.tflite        # ⚠️ 需手动放入（见下）
            ├── scaler.json                # 标准化参数（已就绪）
            ├── har_test_windows.bin       # 600 个原始窗口 float32 小端
            └── har_test_labels.bin        # 600 个标签 int8
```

## 2. 放入模型文件（必须）

转换得到量化模型后，复制到 assets：

```bash
cp ../models/har_cnn_int8.tflite app/src/main/assets/
```

> 若想对比，也可放入 `har_cnn_float32.tflite` 并在 `MainActivity` 中将模型名改为它。

## 3. 构建运行

1. 用 **Android Studio**（建议 Giraffe+/Hedgehog+，AGP 8.x）打开 `android/` 目录。
2. 等待 Gradle 同步（会自动下载 `tensorflow-lite:2.16.1` 等依赖）。
3. 连接 Android 手机（**minSdk 24 / Android 7.0+**）或启动模拟器。
4. 点击 **Run ▶**（或 `Shift+F10`）。
5. App 启动后会：
   - 自动对 600 个内置样本做全量推理，界面显示**准确率**；
   - 点击「随机验证单个样本」按钮，随机抽取一个窗口展示**预测 vs 真值**。

## 4. 关键实现要点

- **模型输入布局**：`(1, 6, 40)`，即 `[batch][通道][时间]`，通道优先。
- **预处理**：与训练端一致，按 `scaler.json` 做 `(x-mean)/std` 的通道级 Z-Score。
- **int8 推理**：`TensorFlow Lite` 解释器内部以 int8 计算，I/O 保持 float32，无需在 Java 端做量化/反量化。
- **资源不压缩**：`build.gradle` 中 `aaptOptions { noCompress "tflite","bin","json" }`，保证可内存映射加载。

## 5. 接入真实 IMU 传感器（实时识别）

`HARClassifier.classify(float[][] window)` 接收 `[40][6]` 的原始窗口（时间×通道）。
对接方式：

```java
// 用 SensorManager 同时监听 TYPE_ACCELEROMETER 与 TYPE_GYROSCOPE
// 以 ~22.2 Hz 缓存最近 40 组 (ax,ay,az,gx,gy,gz) 组成一个窗口
float[][] window = new float[40][6];
// ... 填充 40 帧 ...
String activity = classifier.classify(window);   // 得到活动名称
```

注意加速度含重力分量，与训练数据分布一致即可，无需额外去重力。

## 6. 常见问题

- **黑屏/崩溃**：确认 `assets/har_cnn_int8.tflite` 已放入；确认 `minSdk ≥ 24`。
- **准确率偏低**：确认 `scaler.json` 与训练时一致；确认输入通道顺序为 `ax,ay,az,gx,gy,gz` 且时间在前、通道在后组成 `[40][6]`。
- **构建报 TFLite 版本冲突**：统一使用 `tensorflow-lite:2.16.1`（已锁定），不要混用 `tensorflow-lite-gpu` 旧版。
