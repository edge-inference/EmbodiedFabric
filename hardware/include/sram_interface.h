#ifndef SRAM_INTERFACE_H
#define SRAM_INTERFACE_H

#include <systemc.h>
#include "preprocess_types.h"

// Shared SRAM between preprocessing and accelerator core.
// Preprocessing writes tensors, asserts ready.
// Accelerator reads tensors, writes actions, asserts done.
SC_MODULE(SramInterface) {
    sc_fifo_in<float*>    in_front;
    sc_fifo_in<float*>    in_wrist;
    sc_fifo_in<float*>    in_state;
    sc_fifo_in<int32_t*>  in_tokens;

    sc_out<bool>          preprocess_ready;
    sc_in<bool>           accel_done;
    sc_fifo_out<ActionChunk*> out_action;

    PreprocessedTensors   sram;
    ActionChunk           action_buf;

    void write_tensors();
    void read_actions();

    SC_CTOR(SramInterface) {
        SC_THREAD(write_tensors);
        SC_THREAD(read_actions);
    }
};

#endif
