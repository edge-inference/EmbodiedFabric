#ifndef PREPROCESS_BLOCK_H
#define PREPROCESS_BLOCK_H

#include <systemc.h>
#include "preprocess_types.h"
#include "image_resize.h"
#include "state_normalize.h"
#include "sram_interface.h"

// Top-level preprocessing module.
// Receives raw observations, produces normalized tensors in SRAM.
SC_MODULE(PreprocessBlock) {
    sc_fifo_in<RawObservation*>   in_obs;
    sc_out<bool>                  preprocess_ready;
    sc_in<bool>                   accel_done;
    sc_fifo_out<ActionChunk*>     out_action;

    ImageResize     front_resize;
    ImageResize     wrist_resize;
    StateNormalize  state_norm;
    SramInterface   sram_if;

    // internal FIFOs connecting submodules
    sc_fifo<uint8_t*>  front_raw_fifo;
    sc_fifo<uint8_t*>  wrist_raw_fifo;
    sc_fifo<float*>    front_out_fifo;
    sc_fifo<float*>    wrist_out_fifo;
    sc_fifo<float*>    state_in_fifo;
    sc_fifo<float*>    state_out_fifo;
    sc_fifo<int32_t*>  token_fifo;

    void dispatch();

    SC_HAS_PROCESS(PreprocessBlock);
    PreprocessBlock(sc_module_name name, const NormConstants& nc);
};

#endif
