#include <math.h>
#include "inference.h"
#include "model.h"
#include "preprocess.h"
#include "test_case.h"
#include "esp_log.h"

static const char *TAG_INF = "Test";
#define ASSERT(condition) \
    if (!(condition)) { \
        ESP_LOGE(TAG_INF, "Assertion failed at %s:%d\n", __FILE__, __LINE__); \
        while (1); \
    }

void test_pipeline()
{
    // Test preprocess
    static float audio_buffer[FRAME_SIZE];
    static float x[SPECTRUM_WIDTH * SPECTRUM_HEIGHT];
    float amplitude;
    bool got_features = false;
    for (int i = 0; i < TEST_LENGTH - FRAME_SIZE; i += FRAME_SIZE)
    {
        for (int j = 0; j < FRAME_SIZE; j++)
        {
            audio_buffer[j] = raw_audio[i + j];
        }
        preprocess_put_audio(audio_buffer);
        if (preprocess_get_features(x, &amplitude))
        {
            got_features = true;
            break; // We have a complete feature window
        }
    }
    ASSERT(got_features);
    for (int i = 0; i < SPECTRUM_WIDTH * SPECTRUM_HEIGHT; i++)
    {
        //ESP_LOGI(TAG_INF, "Feature %d: Expected: %f, Actual: %f", i, test_x[i], x[i]);
        ASSERT(fabs(test_x[i] - x[i]) < 1e-4);
    }
    
    // Test quantize
    int8_t *xq = inference_put_features(x);
    for (int i = 0; i < SPECTRUM_WIDTH * SPECTRUM_HEIGHT; i++)
    {
        //ESP_LOGI(TAG_INF, "Quantized Feature %d: Expected: %d, Actual: %d", i, test_xq[i], xq[i]);
        ASSERT(abs(xq[i] - test_xq[i]) <= 1);
    }
    
    // Test prediction
    float prediction[NUM_CLASSES];
    ASSERT(inference_predict(prediction));
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        ESP_LOGI(TAG_INF, "Class %d: Expected: %f, Actual: %f", i, test_prediction[i], prediction[i]);
        ASSERT(fabs(test_prediction[i] - prediction[i]) < 0.1);
    }

    ESP_LOGI(TAG_INF, "*** Pipeline test completed successfully. ***");
}
