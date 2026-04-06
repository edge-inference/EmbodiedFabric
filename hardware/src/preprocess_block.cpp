#include "preprocess_block.h"
#include <cstring>

PreprocessBlock::PreprocessBlock(sc_module_name name, const NormConstants& nc)
    : sc_module(name),
      front_resize("front_resize"),
      wrist_resize("wrist_resize"),
      state_norm("state_norm", nc),
      sram_if("sram_if"),
      front_raw_fifo(1),
      wrist_raw_fifo(1),
      front_out_fifo(1),
      wrist_out_fifo(1),
      state_in_fifo(1),
      state_out_fifo(1),
      token_fifo(1) {

    front_resize.in_img(front_raw_fifo);
    front_resize.out_img(front_out_fifo);

    wrist_resize.in_img(wrist_raw_fifo);
    wrist_resize.out_img(wrist_out_fifo);

    state_norm.in_state(state_in_fifo);
    state_norm.out_state(state_out_fifo);

    sram_if.in_front(front_out_fifo);
    sram_if.in_wrist(wrist_out_fifo);
    sram_if.in_state(state_out_fifo);
    sram_if.in_tokens(token_fifo);
    sram_if.preprocess_ready(preprocess_ready);
    sram_if.accel_done(accel_done);
    sram_if.out_action(out_action);

    SC_THREAD(dispatch);
}

void PreprocessBlock::dispatch() {
    // "Pick up the orange and place it on the plate"
    // tokenized with HuggingFaceTB/SmolVLM2-500M-Video-Instruct, padded to 48
    static int32_t task_tokens[TOKEN_LEN] = {
        31413, 614, 260, 10245, 284, 1379, 357, 335, 260, 6496,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    };

    while (true) {
        RawObservation* obs = in_obs.read();

        // fan out to parallel resize + normalize pipelines
        uint8_t* front_buf = new uint8_t[IMG_RAW_BYTES];
        uint8_t* wrist_buf = new uint8_t[IMG_RAW_BYTES];
        std::memcpy(front_buf, obs->front, IMG_RAW_BYTES);
        std::memcpy(wrist_buf, obs->wrist, IMG_RAW_BYTES);

        float* state_buf = new float[STATE_DIM];
        std::memcpy(state_buf, obs->state, STATE_DIM * sizeof(float));

        int32_t* tok_buf = new int32_t[TOKEN_LEN];
        std::memcpy(tok_buf, task_tokens, TOKEN_LEN * sizeof(int32_t));

        front_raw_fifo.write(front_buf);
        wrist_raw_fifo.write(wrist_buf);
        state_in_fifo.write(state_buf);
        token_fifo.write(tok_buf);

        delete obs;
    }
}
