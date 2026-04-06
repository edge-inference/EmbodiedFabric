#ifndef STATE_NORMALIZE_H
#define STATE_NORMALIZE_H

#include <systemc.h>
#include "preprocess_types.h"

SC_MODULE(StateNormalize) {
    sc_fifo_in<float*>   in_state;
    sc_fifo_out<float*>  out_state;

    NormConstants norm;

    void process();

    SC_HAS_PROCESS(StateNormalize);
    StateNormalize(sc_module_name name, const NormConstants& nc)
        : sc_module(name), norm(nc) {
        SC_THREAD(process);
    }
};

#endif
