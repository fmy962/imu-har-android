package com.example.imuhar;

import android.content.res.AssetManager;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.io.FileInputStream;

/**
 * IMU 人体活动识别 —— Android 端 TFLite 推理封装
 * --------------------------------------------------
 * 模型输入:  float32 (1, 40, 8)   // 时间优先；6 通道 IMU + 2 个零填充通道
 * 模型输出:  float32 (1, 7)        // 7 类 logits
 * 预处理:    与训练端一致的按通道 Z-Score 标准化（参数来自 assets/scaler.json）
 * 说明:      为满足 TFLite int8 CONV_2D 对通道数 4 的倍数的要求，建模通道为 8，
 *           其中前 6 路为真实 IMU（ax,ay,az,gx,gy,gz），后 2 路始终为零。
 */
public class HARClassifier {

    // 类别顺序必须与训练 config.LABELS 一致: C, D, J, T, U, Wa, Wr
    public static final String[] LABEL_NAMES = {
            "Cycling", "Downstair", "Jogging", "Typing",
            "Upstair", "Walking", "Writing"
    };

    private static final int WINDOW = 40;
    private static final int CHANNELS = 8;     // 6 IMU + 2 零填充（满足 int8 约束）
    private static final int IMU_CHANNELS = 6; // 真实 IMU 通道数
    private static final int CLASSES = 7;

    private final Interpreter interpreter;
    private final float[] mean = new float[IMU_CHANNELS];
    private final float[] std = new float[IMU_CHANNELS];

    public HARClassifier(AssetManager assets, String modelName) throws IOException {
        MappedByteBuffer model = loadModelFile(assets, modelName);
        interpreter = new Interpreter(model);
        interpreter.setNumThreads(4);
        loadScaler(assets, "scaler.json");
    }

    private MappedByteBuffer loadModelFile(AssetManager assets, String name) throws IOException {
        android.content.res.AssetFileDescriptor afd = assets.openFd(name);
        FileInputStream fis = new FileInputStream(afd.getFileDescriptor());
        FileChannel channel = fis.getChannel();
        long start = afd.getStartOffset();
        long len = afd.getDeclaredLength();
        return channel.map(FileChannel.MapMode.READ_ONLY, start, len);
    }

    private void loadScaler(AssetManager assets, String name) throws IOException {
        InputStream is = assets.open(name);
        java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
        byte[] buf = new byte[1024];
        int n;
        while ((n = is.read(buf)) != -1) bos.write(buf, 0, n);
        is.close();
        try {
            JSONObject sc = new JSONObject(new String(bos.toByteArray(), "UTF-8"));
            JSONArray m = sc.getJSONArray("mean");
            JSONArray s = sc.getJSONArray("std");
            // scaler 仅含 6 路真实 IMU 的均值/标准差；后 2 个零填充通道无需归一化
            for (int i = 0; i < IMU_CHANNELS; i++) {
                mean[i] = (float) m.getDouble(i);
                std[i] = (float) s.getDouble(i);
                if (std[i] == 0f) std[i] = 1f;
            }
        } catch (Exception e) {
            throw new IOException("parse scaler failed", e);
        }
    }

    /** 对单个窗口(40x6 原始 IMU)推理，返回 7 类概率 */
    public float[] predictProba(float[][] window) {
        ByteBuffer input = ByteBuffer.allocateDirect(1 * WINDOW * CHANNELS * 4)
                .order(ByteOrder.nativeOrder());
        // TFLite 输入布局: [batch][time][channel] = (1,40,8)
        // 前 6 路为真实 IMU，后 2 路为零填充
        for (int t = 0; t < WINDOW; t++) {
            for (int c = 0; c < CHANNELS; c++) {
                float v;
                if (c < IMU_CHANNELS) {
                    v = (window[t][c] - mean[c]) / std[c];
                } else {
                    v = 0f;
                }
                input.putFloat(v);
            }
        }
        input.rewind();

        float[][] output = new float[1][CLASSES];
        interpreter.run(input, output);
        return softmax(output[0]);
    }

    /** 推理并返回活动名称 */
    public String classify(float[][] window) {
        float[] p = predictProba(window);
        return LABEL_NAMES[argmax(p)];
    }

    /** 返回 "名称 (概率)" 便于 UI 展示 */
    public String classifyVerbose(float[][] window) {
        float[] p = predictProba(window);
        int idx = argmax(p);
        return LABEL_NAMES[idx] + "  (" + String.format("%.1f%%", p[idx] * 100) + ")";
    }

    public void close() {
        if (interpreter != null) interpreter.close();
    }

    private static float[] softmax(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) max = Math.max(max, v);
        float sum = 0f;
        float[] o = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            o[i] = (float) Math.exp(x[i] - max);
            sum += o[i];
        }
        for (int i = 0; i < x.length; i++) o[i] /= sum;
        return o;
    }

    private static int argmax(float[] x) {
        int idx = 0;
        float m = x[0];
        for (int i = 1; i < x.length; i++) if (x[i] > m) { m = x[i]; idx = i; }
        return idx;
    }
}
