#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "driver/gpio.h"
#include "esp_log.h"

// Project includes
#include "audio.h"
#include "preprocess.h"
#include "inference.h"
#include "test.h"

// Compile time check preprocessor constants
#if SAMPLE_RATE != AUDIO_SAMPLE_RATE
#error "AUDIO_SAMPLE_RATE must be equal to SAMPLE_RATE from model.h"
#endif
#if FRAME_SIZE != AUDIO_BUFFER_SAMPLES
#error "AUDIO_BUFFER_SAMPLES must be equal to FRAME_SIZE from model.h"
#endif

// Static variables
static float features[SPECTRUM_WIDTH * SPECTRUM_HEIGHT];
static float prediction[NUM_CLASSES];
static const char *TAG_INF = "Inference";

// LED pin
#define LED_PIN GPIO_NUM_21

/**
 * @brief Main setup function.
 */
void setup(void)
{
    // Initialize inference
    if (!inference_init())
    {
        ESP_LOGE(TAG_INF, "Failed to initialize inference!");
        abort();
    }

    // Initialize preprocessing with window stride 24 = 0.384s
    if (!preprocess_init(24))
    {
        ESP_LOGE(TAG_INF, "Failed to initialize preprocessing!");
        abort();
    }

    // Run test case
    // Do this before initializing audio, which starts a thread that may print warnings about missed frames
    test_pipeline();

    // Initialize audio with gain 4.0
    audio_init(4.0f);

    // Initialize LED
    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(LED_PIN, 1); // Turn off LED (active low)
}

/**
 * @brief Main loop function.
 */
void loop(void)
{
    // Obtain audio frame
    float *audio_frame = audio_read();

    // Put and preprocess audio frame
    preprocess_put_audio(audio_frame);
    audio_read_release();

    // Obtain features and run inference if features are ready
    float amplitude;
    if (preprocess_get_features(features, &amplitude))
    {
        // Put features into the interpreter
        inference_put_features(features);

        // Run inference
        if (!inference_predict(prediction))
        {
            ESP_LOGE(TAG_INF, "Failed to invoke interpreter!");
        }
        else
        {
            // Print output
            ESP_LOGI(TAG_INF, "Amplitude: %5.0f, Other: %.2f, Yes: %.2f, No: %.2f", amplitude, prediction[0], prediction[1], prediction[2]);

            // Light up LED if "yes", turn off if "no"
            if (prediction[1] > 0.9f)
            {
                gpio_set_level(LED_PIN, 0); // Turn on LED (active low)
            }
            if (prediction[2] > 0.9f)
            {
                gpio_set_level(LED_PIN, 1); // Turn off LED
            }
        }
    }
}

extern "C" void app_main(void)
{
    setup();
    while (true)
    {
        loop();
    }
}
