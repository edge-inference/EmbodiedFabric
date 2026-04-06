#include "image_resize.h"
#include <algorithm>
#include <cmath>
#include <cstring>

void ImageResize::bilinear_sample(const uint8_t* src, int src_h, int src_w,
                                   float y, float x, uint8_t rgb[3]) {
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = std::min(x0 + 1, src_w - 1);
    int y1 = std::min(y0 + 1, src_h - 1);
    x0 = std::max(x0, 0);
    y0 = std::max(y0, 0);

    float fx = x - std::floor(x);
    float fy = y - std::floor(y);

    for (int c = 0; c < 3; c++) {
        float v00 = src[(y0 * src_w + x0) * 3 + c];
        float v01 = src[(y0 * src_w + x1) * 3 + c];
        float v10 = src[(y1 * src_w + x0) * 3 + c];
        float v11 = src[(y1 * src_w + x1) * 3 + c];
        float val = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy)
                  + v10 * (1 - fx) * fy + v11 * fx * fy;
        rgb[c] = static_cast<uint8_t>(std::min(std::max(val, 0.0f), 255.0f));
    }
}

void ImageResize::process() {
    while (true) {
        uint8_t* raw = in_img.read();

        float scale = static_cast<float>(IMG_TARGET) / std::max(IMG_H_IN, IMG_W_IN);
        int new_h = static_cast<int>(IMG_H_IN * scale);
        int new_w = static_cast<int>(IMG_W_IN * scale);
        int y_off = (IMG_TARGET - new_h) / 2;
        int x_off = (IMG_TARGET - new_w) / 2;

        float* out = new float[IMG_OUT_FLOATS];
        std::memset(out, 0, IMG_OUT_FLOATS * sizeof(float));

        for (int py = 0; py < new_h; py++) {
            for (int px = 0; px < new_w; px++) {
                float src_y = py / scale;
                float src_x = px / scale;
                uint8_t rgb[3];
                bilinear_sample(raw, IMG_H_IN, IMG_W_IN, src_y, src_x, rgb);

                int dy = py + y_off;
                int dx = px + x_off;
                // HWC -> CHW, uint8 -> float32 [0,1]
                for (int c = 0; c < 3; c++) {
                    out[c * IMG_TARGET * IMG_TARGET + dy * IMG_TARGET + dx] =
                        rgb[c] / 255.0f;
                }
            }
        }

        delete[] raw;
        out_img.write(out);
    }
}
