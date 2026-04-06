#include <systemc.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <string>
#include "preprocess_block.h"

#ifndef TV_DIR
#define TV_DIR "tb/test_vectors"
#endif
static const std::string tv_dir = TV_DIR;

bool load_bin_u8(const std::string& path, uint8_t* dst, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return false; }
    f.read(reinterpret_cast<char*>(dst), n);
    return f.good() || f.eof();
}

bool load_bin_f32(const std::string& path, float* dst, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; return false; }
    f.read(reinterpret_cast<char*>(dst), n * sizeof(float));
    return f.good() || f.eof();
}

float max_abs_error(const float* a, const float* b, size_t n) {
    float mx = 0;
    for (size_t i = 0; i < n; i++)
        mx = std::max(mx, std::fabs(a[i] - b[i]));
    return mx;
}

NormConstants load_norm_constants() {
    NormConstants nc;
    load_bin_f32(tv_dir + "/normalization/state_mean.bin", nc.mean, STATE_DIM);
    load_bin_f32(tv_dir + "/normalization/state_std.bin", nc.std, STATE_DIM);
    return nc;
}

SC_MODULE(Testbench) {
    sc_fifo<RawObservation*>  obs_fifo;
    sc_signal<bool>           ready_sig;
    sc_signal<bool>           done_sig;
    sc_fifo<ActionChunk*>     action_fifo;

    PreprocessBlock* dut;

    void run_tests() {
        done_sig.write(false);

        for (int s = 0; s < 3; s++) {
            std::string base = tv_dir + "/sample_" + std::to_string(s);

            auto* obs = new RawObservation;
            if (!load_bin_u8(base + "/front_raw.bin",
                             reinterpret_cast<uint8_t*>(obs->front), IMG_RAW_BYTES)) {
                std::cerr << "FAIL: cannot load front for sample " << s << "\n";
                sc_stop(); return;
            }
            if (!load_bin_u8(base + "/wrist_raw.bin",
                             reinterpret_cast<uint8_t*>(obs->wrist), IMG_RAW_BYTES)) {
                std::cerr << "FAIL: cannot load wrist for sample " << s << "\n";
                sc_stop(); return;
            }
            load_bin_f32(base + "/state_raw.bin", obs->state, STATE_DIM);

            obs_fifo.write(obs);

            wait(ready_sig.posedge_event());

            auto* expected_front = new float[IMG_OUT_FLOATS];
            auto* expected_wrist = new float[IMG_OUT_FLOATS];
            float expected_state[STATE_DIM];

            load_bin_f32(base + "/expected/front_chw_f32.bin", expected_front, IMG_OUT_FLOATS);
            load_bin_f32(base + "/expected/wrist_chw_f32.bin", expected_wrist, IMG_OUT_FLOATS);
            load_bin_f32(base + "/expected/state_norm_f32.bin", expected_state, STATE_DIM);

            float front_err = max_abs_error(
                reinterpret_cast<float*>(dut->sram_if.sram.front),
                expected_front, IMG_OUT_FLOATS);
            float wrist_err = max_abs_error(
                reinterpret_cast<float*>(dut->sram_if.sram.wrist),
                expected_wrist, IMG_OUT_FLOATS);
            float state_err = max_abs_error(
                dut->sram_if.sram.state,
                expected_state, STATE_DIM);

            // C++ bilinear vs OpenCV bilinear differ at subpixel boundaries
            float img_tol = 0.12f;
            float state_tol = 1e-5f;

            std::cout << "Sample " << s
                      << " | front_err=" << front_err
                      << " wrist_err=" << wrist_err
                      << " state_err=" << state_err;

            bool pass = (front_err < img_tol) && (wrist_err < img_tol) && (state_err < state_tol);
            std::cout << (pass ? " PASS" : " FAIL") << "\n";

            delete[] expected_front;
            delete[] expected_wrist;

            done_sig.write(true);
            wait(10, SC_NS);
            done_sig.write(false);
        }

        std::cout << "\nAll samples processed.\n";
        sc_stop();
    }

    SC_CTOR(Testbench) : obs_fifo(1), action_fifo(1) {
        NormConstants nc = load_norm_constants();
        dut = new PreprocessBlock("dut", nc);
        dut->in_obs(obs_fifo);
        dut->preprocess_ready(ready_sig);
        dut->accel_done(done_sig);
        dut->out_action(action_fifo);

        SC_THREAD(run_tests);
    }
};

int sc_main(int, char*[]) {
    Testbench tb("tb");
    sc_start();
    return 0;
}
