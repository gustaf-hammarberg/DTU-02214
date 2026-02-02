#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_pdm.h"
#include "esp_log.h"
#include "audio.h"

// Static variables
static float gain;
static i2s_chan_handle_t rx_handle;
static int16_t audio_buffer[AUDIO_BUFFER_SAMPLES];
static float audio_frame[AUDIO_BUFFER_SAMPLES] = { 0.0 };
static volatile bool audio_frame_ready = false;
static volatile bool audio_frame_in_use = false;
static const char *TAG = "Audio";

// Forward declarations
static void audio_task(void *args);

/**
 * @brief Initialize audio input and start the audio task. Call this once before calling
 * audio_read().
 * 
 * @param gain   Gain to apply to audio samples. 1.0 is no gain.
*/
void audio_init(float gain)
{
    esp_err_t err;

    // Save gain
    ::gain = gain;

    // Allocate an I2S RX channel
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    err = i2s_new_channel(&chan_cfg, NULL, &rx_handle);
    ESP_ERROR_CHECK(err);

    // Initialize the channel for PDM RX
    i2s_pdm_rx_config_t pdm_cfg = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(AUDIO_SAMPLE_RATE),
        .slot_cfg = I2S_PDM_RX_SLOT_PCM_FMT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .clk = GPIO_NUM_42,   // PDM CLK
            .din = GPIO_NUM_41,   // PDM DATA
            .invert_flags = {
                .clk_inv = false,
            },
        },
    };
    err = i2s_channel_init_pdm_rx_mode(rx_handle, &pdm_cfg);
    ESP_ERROR_CHECK(err);

    // Enable channel
    err = i2s_channel_enable(rx_handle);
    ESP_ERROR_CHECK(err);

    // Start audio task
    xTaskCreate(audio_task, "audio_task", 4096, NULL, 1, NULL);
}

/**
 * @brief Read audio frame. This function blocks until an audio frame is ready.
 * @note  The caller assumes ownership over the frame buffer, which prevents further frames to be
 *        written. Always call audio_read_release() when the frame has been processed.
 * 
 * @return Pointer to audio frame buffer of size AUDIO_BUFFER_SAMPLES.
*/
float* audio_read()
{
    // Wait for audio frame to be ready
    while (!audio_frame_ready)
    {
        vTaskDelay(1);
    }
    audio_frame_ready = false;
    audio_frame_in_use = true;
    return audio_frame;
}

/**
 * @brief Release the audio frame previously obtained by audio_read().
*/
void audio_read_release()
{
    audio_frame_in_use = false;
}

static void audio_task(void *args)
{
    esp_err_t err;
    size_t bytes_read;
    size_t i;

    while (true)
    {
        // Read audio data into buffer
        err = i2s_channel_read(rx_handle, audio_buffer, sizeof(audio_buffer), &bytes_read, 1000);
        ESP_ERROR_CHECK(err);
        if (bytes_read != sizeof(audio_buffer))
        {
            ESP_LOGW(TAG, "Read %d bytes instead of %d", bytes_read, sizeof(audio_buffer));

            // Clear rest of buffer
            size_t samples_read = bytes_read / sizeof(audio_buffer[0]);
            if (samples_read < AUDIO_BUFFER_SAMPLES)
            {
                for (i = samples_read; i < AUDIO_BUFFER_SAMPLES; i++)
                {
                    audio_buffer[i] = 0;
                }
            }
        }

        // Prevent audio_read() from returning torn frames
        if (audio_frame_in_use)
        {
            ESP_LOGW(TAG, "Audio frame in use, dropping");
            continue;
        }
        if (audio_frame_ready)
        {
            ESP_LOGW(TAG, "Audio frame not yet consumed, dropping");
            audio_frame_ready = false;
        }
        
        // Convert to float audio frame in range [-32767.0, 32767.0] and apply gain
        for (i = 0; i < AUDIO_BUFFER_SAMPLES; i++)
        {
            float value = ((float)audio_buffer[i]) * gain;
            if (fabs(value) > 32767.0f)
                value = value > 0.0 ? 32767.0 : -32767.0;
            audio_frame[i] = value;
        }

        // Signal audio_read()
        audio_frame_ready = true;
    }
    vTaskDelete(NULL);
}
