bool inference_init();
int8_t* inference_put_features(const float *features);
bool inference_predict(float *prediction);