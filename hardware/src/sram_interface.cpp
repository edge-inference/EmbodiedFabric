#include "sram_interface.h"
#include <cstring>

void SramInterface::write_tensors() {
    while (true) {
        float*   front  = in_front.read();
        float*   wrist  = in_wrist.read();
        float*   state  = in_state.read();
        int32_t* tokens = in_tokens.read();

        std::memcpy(sram.front, front, sizeof(sram.front));
        std::memcpy(sram.wrist, wrist, sizeof(sram.wrist));
        std::memcpy(sram.state, state, sizeof(sram.state));
        std::memcpy(sram.tokens, tokens, sizeof(sram.tokens));

        delete[] front;
        delete[] wrist;
        delete[] state;
        delete[] tokens;

        preprocess_ready.write(true);
        wait(accel_done.posedge_event());
        preprocess_ready.write(false);
    }
}

void SramInterface::read_actions() {
    // stub: accelerator writes action_buf via SRAM
    // in functional model, this is a passthrough
}
