#ifndef PREPROCESS_TYPES_H
#define PREPROCESS_TYPES_H

#include <cstdint>

static constexpr int IMG_H_IN    = 480;
static constexpr int IMG_W_IN    = 640;
static constexpr int IMG_CH      = 3;
static constexpr int IMG_TARGET  = 512;
static constexpr int STATE_DIM   = 6;
static constexpr int ACTION_DIM  = 6;
static constexpr int ACTION_HORIZON = 50;
static constexpr int TOKEN_LEN   = 48;

static constexpr int IMG_RAW_BYTES = IMG_H_IN * IMG_W_IN * IMG_CH;
static constexpr int IMG_OUT_FLOATS = IMG_CH * IMG_TARGET * IMG_TARGET;

struct RawObservation {
    uint8_t  front[IMG_H_IN][IMG_W_IN][IMG_CH];
    uint8_t  wrist[IMG_H_IN][IMG_W_IN][IMG_CH];
    float    state[STATE_DIM];
};

struct PreprocessedTensors {
    float    front[IMG_CH][IMG_TARGET][IMG_TARGET];  // CHW normalized
    float    wrist[IMG_CH][IMG_TARGET][IMG_TARGET];
    float    state[STATE_DIM];
    int32_t  tokens[TOKEN_LEN];
};

struct ActionChunk {
    float data[ACTION_HORIZON][ACTION_DIM];
};

struct NormConstants {
    float mean[STATE_DIM];
    float std[STATE_DIM];
};

#endif
