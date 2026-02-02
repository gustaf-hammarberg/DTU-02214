// ESP includes
#include "esp_camera.h"
#include "esp_log.h"

// Project includes
#include "camera.h"

static const char *TAG = "CAMERA";

static camera_config_t get_camera_config()
{
    camera_config_t config = {};

    // LEDC (for XCLK)
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;

    // Data pins
    config.pin_d0 = 15; // Y2_GPIO_NUM
    config.pin_d1 = 17; // Y3_GPIO_NUM
    config.pin_d2 = 18; // Y4_GPIO_NUM
    config.pin_d3 = 16; // Y5_GPIO_NUM
    config.pin_d4 = 14; // Y6_GPIO_NUM
    config.pin_d5 = 12; // Y7_GPIO_NUM
    config.pin_d6 = 11; // Y8_GPIO_NUM
    config.pin_d7 = 48; // Y9_GPIO_NUM

    // Control pins
    config.pin_xclk  = 10; // XCLK_GPIO_NUM
    config.pin_pclk  = 13; // PCLK_GPIO_NUM
    config.pin_vsync = 38; // VSYNC_GPIO_NUM
    config.pin_href  = 47; // HREF_GPIO_NUM

    // SCCB (I2C) pins
    config.pin_sccb_sda = 40; // SIOD_GPIO_NUM
    config.pin_sccb_scl = 39; // SIOC_GPIO_NUM

    // Power-down and reset (not used on XIAO)
    config.pin_pwdn  = -1; // PWDN_GPIO_NUM
    config.pin_reset = -1; // RESET_GPIO_NUM

    // Clock
    config.xclk_freq_hz = 16000000; // 16 MHz

    // Frame buffer settings
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.fb_count    = 1;
    config.grab_mode   = CAMERA_GRAB_LATEST;

    // Pixel format: RGB565 is efficient to convert to RGB888
    config.pixel_format = PIXFORMAT_RGB565;

    // Resolution: QVGA (320x240)
    config.frame_size = FRAMESIZE_QVGA;

    // JPEG-specific fields (ignored for RGB565 but set to sane defaults)
    config.jpeg_quality = 12;

    return config;
}

/**
 * Initialize the camera.
 * @return true on success, false on failure.
 */
bool camera_init(void)
{
    ESP_LOGI(TAG, "Initializing camera...");

    camera_config_t camera_config = get_camera_config();
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed with error 0x%x.", err);
        return false;
    }

    ESP_LOGI(TAG, "Camera initialized: %dx%d RGB565.", FRAME_W, FRAME_H);

    return true;
}

/**
 * Capture a frame.
 * @param image_buffer Pointer to the output buffer (must be at least FRAME_W x FRAME_H x FRAME_C bytes).
 * @return true on success, false on failure.
 */
bool camera_capture_frame(uint8_t *image_buffer)
{
    // Capture a frame from the camera
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Failed to get frame buffer.");
        return false;
    }

    // Verify frame size and format
    if (fb->width != FRAME_W || fb->height != FRAME_H || fb->format != PIXFORMAT_RGB565) {
        ESP_LOGE(TAG, "Unexpected frame format: %dx%d, format %d.", fb->width, fb->height, fb->format);
        esp_camera_fb_return(fb);
        return false;
    }

    // Verify buffer size
    if (fb->len < FRAME_W * FRAME_H * FRAME_C) {
        ESP_LOGE(TAG, "Frame buffer too small: %d bytes, expected at least %d bytes.", fb->len, FRAME_W * FRAME_H * FRAME_C);
        esp_camera_fb_return(fb);
        return false;
    }

    // Convert the frame buffer to a model input tensor
    memcpy(image_buffer, fb->buf, FRAME_W * FRAME_H * FRAME_C);

    // Return the frame buffer to the driver
    esp_camera_fb_return(fb);

    return true;
}
