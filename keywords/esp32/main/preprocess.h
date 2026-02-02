#pragma once

#include <stddef.h>
#include "model.h"

bool preprocess_init(size_t stride);
void preprocess_put_audio(const float* audio_frame);
bool preprocess_get_features(float* features, float* amplitude);