#ifndef IMAGE_RESIZE_H
#define IMAGE_RESIZE_H

#include <systemc.h>
#include "preprocess_types.h"

SC_MODULE(ImageResize) {
    sc_fifo_in<uint8_t*>  in_img;   // raw HWC uint8 (480x640x3)
    sc_fifo_out<float*>   out_img;  // normalized CHW float32 (3x512x512)

    void process();

    SC_CTOR(ImageResize) {
        SC_THREAD(process);
    }

private:
    // bilinear interpolation of a single pixel
    void bilinear_sample(const uint8_t* src, int src_h, int src_w,
                         float y, float x, uint8_t rgb[3]);
};

#endif
