package com.example.imuhar;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

/**
 * Android 端推理测试界面
 * --------------------------------------------------
 * 启动即对 assets 内自带的测试样本（har_test_windows.bin / har_test_labels.bin）
 * 进行全量推理，统计准确率并展示；点击按钮随机抽一个样本显示预测 vs 真值。
 */
public class MainActivity extends AppCompatActivity {

    private static final int WINDOW = 40;
    private static final int CHANNELS = 6;
    private static final int PER_WINDOW = WINDOW * CHANNELS; // 240

    private HARClassifier classifier;
    private float[] windows;   // N * 240 原始 IMU（小端 float32）
    private byte[] labels;     // N 个 int8 标签索引
    private int nSamples;
    private final Random rng = new Random(1);
    private TextView tvLog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tvLog = findViewById(R.id.tvLog);
        Button btnRandom = findViewById(R.id.btnRandom);

        try {
            AssetManager am = getAssets();
            classifier = new HARClassifier(am, "har_cnn_int8.tflite");
            windows = readFloats(am, "har_test_windows.bin");
            labels = readBytes(am, "har_test_labels.bin");
            nSamples = labels.length;

            runFullTest();
        } catch (IOException e) {
            log("加载模型/资源失败: " + e.getMessage());
            e.printStackTrace();
        }

        btnRandom.setOnClickListener(v -> runRandomSample());
    }

    /** 全量推理测试：统计准确率 */
    private void runFullTest() {
        int correct = 0;
        for (int i = 0; i < nSamples; i++) {
            float[][] w = slice(i);
            String pred = classifier.classify(w);
            String truth = HARClassifier.LABEL_NAMES[labels[i] & 0xFF];
            if (pred.equals(truth)) correct++;
        }
        float acc = correct / (float) nSamples * 100f;
        log("===== Android 端全量推理测试 =====");
        log("样本数: " + nSamples);
        log("准确率: " + String.format("%.2f%%", acc));
        log("模型: har_cnn_int8.tflite (int8 量化)");
        log("-------------------------------");
        log("点击按钮随机验证单个样本");
    }

    /** 随机抽一个样本，展示预测与真值 */
    private void runRandomSample() {
        int i = rng.nextInt(nSamples);
        float[][] w = slice(i);
        String pred = classifier.classify(w);
        String truth = HARClassifier.LABEL_NAMES[labels[i] & 0xFF];
        boolean ok = pred.equals(truth);
        log(String.format("[样本 #%d] 预测=%s  真值=%s  %s",
                i, pred, truth, ok ? "✓" : "✗"));
    }

    /** 从扁平数组中取出第 i 个窗口 -> [40][6] */
    private float[][] slice(int i) {
        float[][] w = new float[WINDOW][CHANNELS];
        int base = i * PER_WINDOW;
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

    private void log(String s) {
        tvLog.append(s + "\n");
    }
}
