#include "state_normalize.h"
#include <cmath>

void StateNormalize::process() {
    static constexpr float EPS = 1e-8f;

    while (true) {
        float* raw = in_state.read();
        float* out = new float[STATE_DIM];

        for (int i = 0; i < STATE_DIM; i++) {
            float s = (std::fabs(norm.std[i]) < EPS) ? EPS : norm.std[i];
            out[i] = (raw[i] - norm.mean[i]) / s;
        }

        delete[] raw;
        out_state.write(out);
    }
}
