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
 * IMU 人体活动识别 —— Android 端实时检测（v1.2）
 * --------------------------------------------------
 * 训练规格（与训练端严格一致）：
 *   加速度 ax/ay/az 单位 g（含重力），陀螺仪 gx/gy/gz 单位 °/s
 *   采样率 ≈22.2 Hz、窗口 40 样本 = 1.8 s；按通道 Z-Score（scaler.json）
 *   模型输入 (1,40,8) 时间优先；前 6 路真实 IMU，后 2 路零填充
 *
 * 本版改进（针对“腕戴模型放到手持手机上效果差”）：
 *   1) 开局重力对齐校准：静止 2 s 估计重力方向，把设备坐标系旋转到
 *      重力恒在 +Z 轴的标准姿态，缓解“任意握持”导致的轴系差异。
 *   2) 输出 EMA 平滑 + 最近帧多数投票，结果不再单帧乱跳。
 *   3) 静止检测 + 低置信度判定：手不动/不确定时不再瞎报。
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
    private static final long INFER_PERIOD = 250;       // 推理节拍(ms)

    // ---------- 校准 / 后处理参数 ----------
    private static final long CALIB_MS = 2000;          // 开局静止校准时长
    private static final float EMA_ALPHA = 0.30f;       // 概率指数平滑系数
    private static final float CONF_THRESHOLD = 0.30f;  // 低于此置信度显示“不确定”
    private static final float STILL_ENERGY = 0.05f;    // 动态加速度能量(去重力)低于此判“静止”(RMS≈0.22g)
    private static final float STILL_GYRO = 2.0f;       // 陀螺仪能量(°/s²)低于此判“静止”
    private static final int VOTE_WINDOW = 6;           // 投票窗口帧数

    // 状态机
    private static final int ST_IDLE = 0;
    private static final int ST_CALIB = 1;
    private static final int ST_DETECT = 2;
    private int state = ST_IDLE;

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

    // 校准累加
    private long calibStart = 0;
    private float[] calibSum = new float[3];
    private int calibCount = 0;

    // 重力对齐旋转（Rodrigues）：把设备重力向量旋转到 (0,0,1)
    private float[] rotAxis = {0, 0, 1};
    private float rotSin = 0f, rotCos = 1f;
    private boolean aligned = false;

    // 后处理状态
    private float[] probEma = null;
    private final ArrayList<Integer> voteBuf = new ArrayList<>();

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

        tvStatus.setText("模型已就绪（int8 量化，测试集准确率 96.15%）。点「开始检测」：先用你打算握持的姿势静止 2 秒校准，再实时识别。");
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
        if (state == ST_IDLE) startDetect();
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
        calibSum[0] = calibSum[1] = calibSum[2] = 0f;
        calibCount = 0;
        aligned = false;
        probEma = null;
        voteBuf.clear();
        calibStart = System.currentTimeMillis();
        state = ST_CALIB;
        detecting = true;
        // GAME 档位约 50 Hz，足够重采样到 22.2 Hz
        sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_GAME);
        sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_GAME);
        btnToggle.setText("校准中…");
        btnToggle.setEnabled(false);
        tvStatus.setText("【校准中】请保持你打算使用的握持姿势，手机尽量静止 2 秒…");
        inferHandler.postDelayed(inferRunnable, INFER_PERIOD);
    }

    private void stopDetect() {
        detecting = false;
        state = ST_IDLE;
        sensorManager.unregisterListener(this);
        inferHandler.removeCallbacks(inferRunnable);
        btnToggle.setText("开始检测");
        btnToggle.setEnabled(true);
        tvStatus.setText("已停止。");
        tvResult.setText("—");
    }

    private final Runnable inferRunnable = new Runnable() {
        @Override
        public void run() {
            if (detecting) {
                if (state == ST_CALIB) {
                    // 校准尚未结束，继续等待
                    if (System.currentTimeMillis() - calibStart >= CALIB_MS && calibCount > 5) {
                        finishCalibration();
                    }
                } else if (state == ST_DETECT) {
                    runInference();
                }
                inferHandler.postDelayed(this, INFER_PERIOD);
            }
        }
    };

    @Override
    public void onSensorChanged(SensorEvent event) {
        long t = System.currentTimeMillis();
        float[] v = new float[]{event.values[0], event.values[1], event.values[2]};
        if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            lastAccel = v;
            if (state == ST_CALIB) {
                calibSum[0] += v[0]; calibSum[1] += v[1]; calibSum[2] += v[2];
                calibCount++;
            }
        } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            lastGyro = v;
        } else {
            return;
        }
        if (state == ST_DETECT && lastAccel != null && lastGyro != null) {
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

    /** 用校准期间的平均加速度估计重力方向，构造对齐旋转 */
    private void finishCalibration() {
        float[] g = new float[]{calibSum[0] / calibCount, calibSum[1] / calibCount, calibSum[2] / calibCount};
        float len = (float) Math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        if (len < 1e-3f) {
            // 退化：不旋转
            rotAxis = new float[]{0, 0, 1}; rotSin = 0f; rotCos = 1f; aligned = false;
        } else {
            g[0] /= len; g[1] /= len; g[2] /= len;
            computeAlignRotation(g, new float[]{0, 0, 1});
            aligned = true;
        }
        state = ST_DETECT;
        btnToggle.setText("停止检测");
        btnToggle.setEnabled(true);
        rawBuf.clear();
        tsBuf.clear();
        tvStatus.setText("✅ 校准完成，开始实时检测（请做出动作）。缓冲中…");
    }

    /** Rodrigues：构造把向量 g 旋转到 t 的最小旋转，保存 axis/sin/cos 供 rotateVec 使用 */
    private void computeAlignRotation(float[] g, float[] t) {
        float[] axis = new float[]{
                g[1] * t[2] - g[2] * t[1],
                g[2] * t[0] - g[0] * t[2],
                g[0] * t[1] - g[1] * t[0]
        };
        float sin = (float) Math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        float cos = g[0] * t[0] + g[1] * t[1] + g[2] * t[2];
        if (sin < 1e-4f) {
            if (cos < 0) {
                // 反平行：绕任意垂直轴转 180°
                if (Math.abs(g[0]) < 0.9f) axis = new float[]{1, 0, 0};
                else axis = new float[]{0, 1, 0};
                rotAxis = axis; rotSin = 0f; rotCos = -1f;
            } else {
                rotAxis = new float[]{0, 0, 1}; rotSin = 0f; rotCos = 1f; // 同向，不旋转
            }
        } else {
            axis[0] /= sin; axis[1] /= sin; axis[2] /= sin;
            rotAxis = axis; rotSin = sin; rotCos = cos;
        }
    }

    /** Rodrigues 旋转一个 3 维向量（旋转前 3 与后 3 共用同一坐标系变换） */
    private void rotateVec(float[] x) {
        // v_rot = v*cos + (axis × v)*sin + axis*(axis·v)*(1-cos)
        float ax = rotAxis[0], ay = rotAxis[1], az = rotAxis[2];
        float crossX = ay * x[2] - az * x[1];
        float crossY = az * x[0] - ax * x[2];
        float crossZ = ax * x[1] - ay * x[0];
        float dot = ax * x[0] + ay * x[1] + az * x[2];
        float s = rotSin;
        float m = (1f - rotCos) * dot;
        x[0] = x[0] * rotCos + crossX * s + ax * m;
        x[1] = x[1] * rotCos + crossY * s + ay * m;
        x[2] = x[2] * rotCos + crossZ * s + az * m;
    }

    /** 取最近 1.8 s 样本，重采样为 40 个等间隔点，重力对齐 + 单位换算后推理 */
    private void runInference() {
        long now = System.currentTimeMillis();
        float[][] window;
        int count;
        synchronized (rawBuf) {
            count = rawBuf.size();
            if (count < 8) {
                tvStatus.setText("检测中… 样本数=" + count + "（需要更多数据）");
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
                tvStatus.setText("检测中… 有效窗口样本=" + seg.size());
                return;
            }
            window = resample(seg, segTs, start);
        }

        // 重力对齐（仅旋转，不改量纲） + 单位换算（m/s² -> g，rad/s -> °/s）
        for (int t = 0; t < WINDOW; t++) {
            if (aligned) rotateVec(window[t]);   // 旋转前 3（加速度）与后 3（陀螺仪）
            for (int c = 0; c < 3; c++) window[t][c] /= GRAVITY;
            for (int c = 3; c < 6; c++) window[t][c] *= RAD2DEG;
        }

        float[] prob = classifier.predictProba(window);
        int idx = argmax(prob);

        // EMA 平滑
        if (probEma == null) probEma = prob.clone();
        else for (int i = 0; i < prob.length; i++)
            probEma[i] = EMA_ALPHA * prob[i] + (1 - EMA_ALPHA) * probEma[i];

        int smoothIdx = argmax(probEma);

        // 静止检测：对齐后重力在 +Z，动态加速度为前 2 轴 (ax,ay)；陀螺仪能量(°/s²)
        float dynAcc = 0f;
        for (int t = 0; t < WINDOW; t++)
            dynAcc += window[t][0] * window[t][0] + window[t][1] * window[t][1];
        dynAcc /= WINDOW;
        float gyroE = 0f;
        for (int t = 0; t < WINDOW; t++)
            for (int c = 3; c < 6; c++) gyroE += window[t][c] * window[t][c];
        gyroE /= WINDOW;
        boolean still = (dynAcc < STILL_ENERGY) && (gyroE < STILL_GYRO);

        // 投票（最近 VOTE_WINDOW 帧的平滑 argmax）
        voteBuf.add(smoothIdx);
        while (voteBuf.size() > VOTE_WINDOW) voteBuf.remove(0);
        int voteIdx = majority(voteBuf);
        float voteConf = probEma[voteIdx];

        // 结果展示
        if (still) {
            tvResult.setText("静止");
        } else if (voteConf < CONF_THRESHOLD) {
            tvResult.setText("活动中（不确定）");
        } else {
            tvResult.setText(HARClassifier.LABEL_NAMES[voteIdx] + " / " + LABEL_CN[voteIdx]
                    + "  " + String.format("%.1f%%", voteConf * 100));
        }
        updateBars(probEma);
        tvStatus.setText("实时检测中… 缓冲=" + count + "  | 加速度能量="
                + String.format("%.3f", dynAcc) + (aligned ? "  [已对齐]" : "  [未对齐]"));
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

    private int majority(ArrayList<Integer> list) {
        int[] cnt = new int[HARClassifier.LABEL_NAMES.length];
        for (int v : list) cnt[v]++;
        int best = list.get(0), bestN = 0;
        for (int i = 0; i < cnt.length; i++) if (cnt[i] > bestN) { bestN = cnt[i]; best = i; }
        return best;
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
