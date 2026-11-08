# 方式 B：生成 APK 并手动安装到手机

> 适用场景：想拿一个 `.apk` 文件，通过微信/数据线/U盘传到手机安装，或发给别人安装。
> 前提：本机已安装 **Android Studio + Android SDK**（见下文"前置：安装 SDK"）。
> 本工程：`minSdk 24`（Android 7.0+）、无联网/存储权限、纯本地 TFLite 推理。

---

## 前置：安装 Android Studio 与 SDK（仅首次需要）

1. 下载：https://developer.android.google.cn/studio （国内走 google.cn 更快）
2. 安装时一路默认；首次启动的 Setup 向导会自动下载 **Android SDK**。
   - 联网不畅时：开 Clash 后，在 Studio 里 `Settings → System Settings → HTTP Proxy → Manual`，填 `127.0.0.1:7897`。
   - 必备组件（`Settings → SDK Manager`）：**Android 14 (API 34) 平台**、**Build-Tools 34.0.0**、**Platform-Tools**。
   - 默认 SDK 路径：`C:\Users\<你的用户名>\AppData\Local\Android\Sdk`
3. 确认：打开 `CMD` 执行 `where adb` 能找到 `platform-tools\adb.exe` 即成功。

---

## 步骤 1：用 Android Studio 打开工程

- `Open` → 选择本工程目录 `imu_har/android/`
- 首次打开会 **Gradle 同步**（底部状态栏），自动下载：
  - Gradle 8.2.2（wrapper 自动拉取）
  - `org.tensorflow:tensorflow-lite:2.16.1` 等依赖（来自 google() / mavenCentral()）
  - 国内慢的话，在根 `build.gradle` 的 `repositories` 加阿里云镜像：
    ```gradle
    maven { url 'https://maven.aliyun.com/repository/google' }
    maven { url 'https://maven.aliyun.com/repository/public' }
    ```
- 同步成功（无红色报错）后再继续。

## 步骤 2：生成 Debug APK

1. 顶部菜单：`Build → Build Bundle(s) / APK(s) → Build APK(s)`
2. 等待编译（看底部 `Build` 窗口 / 右上角进度）。约几十秒到几分钟。
3. 编译完成，右下角弹 **"APK(s) generated successfully"**，点其中的 **locate** 直接打开所在文件夹。
   - 或手动去：`android/app/build/outputs/apk/debug/app-debug.apk`
4. `app-debug.apk` 已自带 **debug 签名**，可直接安装，无需自己签名。

## 步骤 3：传到手机并安装

1. 把 `app-debug.apk` 传到手机：
   - 数据线复制、微信"文件传输助手"、或 U 盘均可。
2. 手机里点开该 APK：
   - 首次会提示 **"允许来自此来源的应用" / "未知来源"**，点允许/继续。
   - 按提示完成安装。
3. 桌面出现 **"IMU HAR"** 图标，点开即启动 → 自动全量推理 600 样本并显示准确率。

> 想发正式版（非 debug、可上架/更干净）：`Build → Generate Signed Bundle / APK → APK`，用自己的密钥库签名生成 release APK。本工程 `release` 已设 `minifyEnabled false`（不混淆），可直接用。

---

## 常见问题排查

| 现象 | 原因 / 解决 |
|---|---|
| 找不到 `app-debug.apk` | 确认步骤 2 编译成功；路径在 `app/build/outputs/apk/debug/` |
| `Build` 报错 "SDK location not found" | 没装 SDK；或工程缺 `local.properties`。在 Studio `SDK Manager` 装好，Studio 会自动写 `local.properties` |
| Gradle 同步一直转圈/失败 | 依赖下载慢/被墙。开 Clash 代理，或加阿里云镜像（见步骤 1） |
| 编译报 `Could not find tensorflow-lite:2.16.1` | 仓库没加 `google()`；根 `build.gradle` 的 `allprojects.repositories` 需含 `google()` 与 `mavenCentral()`（本工程已含） |
| 手机安装提示"解析包错误" | APK 下载/传输损坏，重新传；或手机系统低于 Android 7.0（本工程 minSdk 24） |
| 打开 App 闪退 | 确认 `app/src/main/assets/` 下有 4 个文件：`har_cnn_int8.tflite`、`scaler.json`、`har_test_windows.bin`、`har_test_labels.bin`；上一轮已修复 `HARClassifier` 的 scaler 读取崩溃 |

## 关于 assets 不被压缩

`app/build.gradle` 已配置：
```gradle
aaptOptions { noCompress "tflite", "bin", "json" }
```
保证 `har_cnn_int8.tflite` 等能被 `MappedByteBuffer` 内存映射加载（否则压缩后无法 mmap，会加载失败）。**不要删掉这段配置。**
