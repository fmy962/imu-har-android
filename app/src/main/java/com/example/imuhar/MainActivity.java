package com.example.imuhar;

import android.content.res.AssetManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Random;

/**
 * IMU 人体活动识别 —— Android 端实时检测
 * --------------------------------------------------
 * 启动后用手机「加速度计 + 陀螺仪」实时采集，按训练规格
 * （采样率 ≈22.2 Hz、窗口 40 样本 = 1.8 s）做时间重采样，
 * 单位换算（m/s²→g、rad/s→°/s）后送入 TFLite int8 模型推理，
 * 实时显示当前活动与 7 类概率。
 *
 * 训练规格（务必与训练端一致）：
 *   加速度 ax/ay/az 单位 g（含重力），陀螺仪 gx/gy/gz 单位 °/s
 *   按通道 Z-Score 标准化（参数见 assets/scaler.json）
 *   模型输入 (1,40,8) 时间优先；前 6 路为真实 IMU，后 2 路零填充
 */
public class MainActivity extends AppCompatActivity implements SensorEventListener {

    // ---------- 训练规格常量 ----------
    private static final float SAMPLE_HZ = 22.222f;     // 1000 / 45 ms
    private static final long WINDOW_MS = 1800;         // 40 × 45 ms = 1.8 s
    private static final long STEP_MS = 45;             // 重采样间隔
    private static final int WINDOW = 40;
    private static final int CHANNELS = 6;
    private static final float GRAVITY = 9.80665f;      // m/s² -> g
    private static final float RAD2DEG = 57.2957795f;   // rad/s -> °/s
    private static final long INFER_PERIOD = 300;       // 推理节拍(ms)

    // 类别中文名（与 HARClassifier.LABEL_NAMES 顺序一致）
    private static final String[] LABEL_CN = {
            "骑车", "下楼梯", "慢跑", "打字", "上楼梯", "走路", "写字"
    };

    private HARClassifier classifier;
    private SensorManager sensorManager;
    private Sensor accel, gyro;

    // 原始缓冲（手机单位：m/s² & rad/s），带时间戳(ms)
    private final ArrayList<float[]> rawBuf = new ArrayList<>();
    private final ArrayList<Long> tsBuf = new ArrayList<>();
    private float[] lastAccel = null, lastGyro = null;

    private boolean detecting = false;
    private final Handler inferHandler = new Handler(Looper.getMainLooper());

    // 离线测试资源
    private float[] windows;
    private byte[] labels;
    private int nSamples;
    private final Random rng = new Random(1);

    // UI
    private TextView tvStatus, tvResult, tvLog;
    private LinearLayout probContainer;
    private Button btnToggle, btnTest;
    private final ProgressBar[] probBars = new ProgressBar[HARClassifier.LABEL_NAMES.length];
    private final TextView[] probPct = new TextView[HARClassifier.LABEL_NAMES.length];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tvStatus = findViewById(R.id.tvStatus);
        tvResult = findViewById(R.id.tvResult);
        tvLog = findViewById(R.id.tvLog);
        probContainer = findViewById(R.id.probContainer);
        btnToggle = findViewById(R.id.btnToggle);
        btnTest = findViewById(R.id.btnTest);

        buildProbBars();

        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        try {
            classifier = new HARClassifier(getAssets(), "har_cnn_int8.tflite");
            windows = readFloats(getAssets(), "har_test_windows.bin");
            labels = readBytes(getAssets(), "har_test_labels.bin");
            nSamples = labels.length;
        } catch (IOException e) {
            log("加载模型/资源失败: " + e.getMessage());
            e.printStackTrace();
        }

        btnToggle.setOnClickListener(v -> toggleDetect());
        btnTest.setOnClickListener(v -> runFullTest());

        tvStatus.setText("模型已就绪（int8 量化，测试集准确率 96.15%）。点击「开始检测」用手机传感器实时识别。");
        tvResult.setText("—");
    }

    /** 动态构建 7 类概率条 */
    private void buildProbBars() {
        probContainer.removeAllViews();
        for (int i = 0; i < HARClassifier.LABEL_NAMES.length; i++) {
            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.HORIZONTAL);
            LinearLayout.LayoutParams rp = new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
            rp.setMargins(0, 4, 0, 4);
            row.setLayoutParams(rp);

            TextView lab = new TextView(this);
            lab.setText(HARClassifier.LABEL_NAMES[i]);
            lab.setTextSize(13);
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(120, LinearLayout.LayoutParams.WRAP_CONTENT);
            lab.setLayoutParams(lp);

            ProgressBar bar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
            LinearLayout.LayoutParams bp = new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
            bar.setLayoutParams(bp);
            bar.setMax(1000);
            bar.setProgress(0);

            TextView pct = new TextView(this);
            pct.setText("0%");
            pct.setTextSize(13);
            LinearLayout.LayoutParams pp = new LinearLayout.LayoutParams(70, LinearLayout.LayoutParams.WRAP_CONTENT);
            pct.setLayoutParams(pp);

            row.addView(lab);
            row.addView(bar);
            row.addView(pct);
            probContainer.addView(row);
            probBars[i] = bar;
            probPct[i] = pct;
        }
    }

    private void toggleDetect() {
        if (!detecting) startDetect();
        else stopDetect();
    }

    private void startDetect() {
        if (classifier == null) {
            tvStatus.setText("⚠️ 模型未加载，无法检测");
            return;
        }
        if (accel == null || gyro == null) {
            tvStatus.setText("⚠️ 该设备缺少加速度计或陀螺仪传感器");
            return;
        }
        rawBuf.clear();
        tsBuf.clear();
        lastAccel = null;
        lastGyro = null;
        // GAME 档位约 50 Hz 采集，再由时间重采样对齐 22.2 Hz
        sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_GAME);
        sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_GAME);
        detecting = true;
        btnToggle.setText("停止检测");
        tvStatus.setText("采集中… 请做出动作（走路/慢跑/上楼/打字/写字/骑车）");
        inferHandler.postDelayed(inferRunnable, INFER_PERIOD);
    }

    private void stopDetect() {
        detecting = false;
        sensorManager.unregisterListener(this);
        inferHandler.removeCallbacks(inferRunnable);
        btnToggle.setText("开始检测");
        tvStatus.setText("已停止。");
    }

    private final Runnable inferRunnable = new Runnable() {
        @Override
        public void run() {
            if (detecting) {
                runInference();
                inferHandler.postDelayed(this, INFER_PERIOD);
            }
        }
    };

    @Override
    public void onSensorChanged(SensorEvent event) {
        long t = System.currentTimeMillis();
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            lastAccel = new float[]{event.values[0], event.values[1], event.values[2]};
        } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            lastGyro = new float[]{event.values[0], event.values[1], event.values[2]};
        } else {
            return;
        }
        if (lastAccel != null && lastGyro != null) {
            float[] s = new float[]{
                    lastAccel[0], lastAccel[1], lastAccel[2],
                    lastGyro[0], lastGyro[1], lastGyro[2]
            };
            synchronized (rawBuf) {
                rawBuf.add(s);
                tsBuf.add(t);
                long cutoff = t - 3000; // 仅保留最近 3 秒
                while (!tsBuf.isEmpty() && tsBuf.get(0) < cutoff) {
                    tsBuf.remove(0);
                    rawBuf.remove(0);
                }
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // 暂不使用
    }

    /** 取最近 1.8 s 样本，重采样为 40 个等间隔点，单位换算后推理 */
    private void runInference() {
        long now = System.currentTimeMillis();
        float[][] window;
        int count;
        synchronized (rawBuf) {
            count = rawBuf.size();
            if (count < 8) {
                tvStatus.setText("采集中… 样本数=" + count + "（需要更多数据）");
                return;
            }
            ArrayList<float[]> seg = new ArrayList<>();
            ArrayList<Long> segTs = new ArrayList<>();
            long start = now - WINDOW_MS;
            for (int i = 0; i < tsBuf.size(); i++) {
                if (tsBuf.get(i) >= start) {
                    seg.add(rawBuf.get(i));
                    segTs.add(tsBuf.get(i));
                }
            }
            if (seg.size() < 2) {
                tvStatus.setText("采集中… 有效窗口样本=" + seg.size());
                return;
            }
            window = resample(seg, segTs, start);
        }

        // 单位换算：加速度 m/s² -> g；陀螺仪 rad/s -> °/s（与训练量纲一致）
        for (int t = 0; t < WINDOW; t++) {
            for (int c = 0; c < 3; c++) window[t][c] /= GRAVITY;
            for (int c = 3; c < 6; c++) window[t][c] *= RAD2DEG;
        }

        float[] prob = classifier.predictProba(window);
        int idx = argmax(prob);
        tvResult.setText(HARClassifier.LABEL_NAMES[idx] + " / " + LABEL_CN[idx]
                + "  " + String.format("%.1f%%", prob[idx] * 100));
        updateBars(prob);
        tvStatus.setText("实时检测中… 缓冲样本=" + count + "（目标 22.2 Hz × 1.8 s）");
    }

    /** 将最近 WINDOW_MS 的样本线性插值到 40 个等间隔点（45 ms） */
    private float[][] resample(ArrayList<float[]> seg, ArrayList<Long> segTs, long start) {
        float[][] out = new float[WINDOW][CHANNELS];
        int n = seg.size();
        for (int k = 0; k < WINDOW; k++) {
            long tk = start + k * STEP_MS;
            if (tk <= segTs.get(0)) {
                System.arraycopy(seg.get(0), 0, out[k], 0, CHANNELS);
            } else if (tk >= segTs.get(n - 1)) {
                System.arraycopy(seg.get(n - 1), 0, out[k], 0, CHANNELS);
            } else {
                int j = 1;
                while (j < n && segTs.get(j) < tk) j++;
                long t0 = segTs.get(j - 1), t1 = segTs.get(j);
                float a = (t1 == t0) ? 1f : (float) (tk - t0) / (t1 - t0);
                float[] s0 = seg.get(j - 1), s1 = seg.get(j);
                for (int c = 0; c < CHANNELS; c++) {
                    out[k][c] = s0[c] + a * (s1[c] - s0[c]);
                }
            }
        }
        return out;
    }

    private void updateBars(float[] prob) {
        for (int i = 0; i < prob.length; i++) {
            probBars[i].setProgress(Math.min(1000, (int) (prob[i] * 1000)));
            probPct[i].setText(String.format("%d%%", (int) (prob[i] * 100)));
        }
    }

    // ---------- 离线测试（验证模型/预处理正确性，原功能保留） ----------
    private void runFullTest() {
        if (classifier == null) {
            log("模型未加载");
            return;
        }
        int correct = 0;
        for (int i = 0; i < nSamples; i++) {
            float[][] w = slice(i);
            String pred = classifier.classify(w);
            String truth = HARClassifier.LABEL_NAMES[labels[i] & 0xFF];
            if (pred.equals(truth)) correct++;
        }
        float acc = correct / (float) nSamples * 100f;
        log("===== 内置样本离线测试 =====");
        log("样本数: " + nSamples + "  准确率: " + String.format("%.2f%%", acc));
        log("（内置样本为原数据集导出，验证模型与预处理一致性）");
        log("点击「开始检测」切换到真实传感器实时识别。");
    }

    private float[][] slice(int i) {
        float[][] w = new float[WINDOW][CHANNELS];
        int base = i * WINDOW * CHANNELS;
        for (int t = 0; t < WINDOW; t++) {
            for (int c = 0; c < CHANNELS; c++) {
                w[t][c] = windows[base + t * CHANNELS + c];
            }
        }
        return w;
    }

    private float[] readFloats(AssetManager am, String name) throws IOException {
        byte[] b = readBytes(am, name);
        ByteBuffer bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
        float[] f = new float[b.length / 4];
        for (int i = 0; i < f.length; i++) f[i] = bb.getFloat();
        return f;
    }

    private byte[] readBytes(AssetManager am, String name) throws IOException {
        InputStream is = am.open(name);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[4096];
        int n;
        while ((n = is.read(buf)) != -1) bos.write(buf, 0, n);
        is.close();
        return bos.toByteArray();
    }

    private static int argmax(float[] x) {
        int idx = 0;
        float m = x[0];
        for (int i = 1; i < x.length; i++) if (x[i] > m) {
            m = x[i];
            idx = i;
        }
        return idx;
    }

    private void log(String s) {
        tvLog.append(s + "\n");
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (detecting) stopDetect();
    }
}
