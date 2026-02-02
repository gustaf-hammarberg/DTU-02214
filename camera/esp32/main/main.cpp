#include <cstdio>
#include <cstdint>
#include <cstring>

// ESP includes
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "driver/usb_serial_jtag.h"

// Project includes
#include "camera.h"

// Static constants and variables
static constexpr size_t CHUNK_SIZE = 256;
static const char* FRAME_PREAMBLE = "\n===FRAME===\n";
static uint8_t image_buffer[FRAME_W * FRAME_H * FRAME_C];

void setup()
{
    // Initialize NVS (required by some drivers)
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    // Initialize camera
    if (!camera_init()) {
        abort();
    }

    // Initialize USB serial
    usb_serial_jtag_driver_config_t cfg = {
        .tx_buffer_size = 512,
        .rx_buffer_size = 512,
    };
    err = usb_serial_jtag_driver_install(&cfg);
    ESP_ERROR_CHECK(err);

    // Wait for incoming S on serial port
    printf("Send 'S' to start.\n");
    char c;
    do {
        int r = usb_serial_jtag_read_bytes(&c, 1, portMAX_DELAY);
        if (r < 0) {
            abort();
        }
    } while (c != 'S');
}

void loop(void)
{
    // Capture frame into tensor
    if (camera_capture_frame(image_buffer)) {
        // Send preamble
        usb_serial_jtag_write_bytes(FRAME_PREAMBLE, strlen(FRAME_PREAMBLE), pdMS_TO_TICKS(1000));

        // Send image over USB console
        // Note that usb_serial_jtag_write_bytes() may fail if writing too many bytes at once, so it's necessary to
        // send in chunks.
        size_t frame_size = sizeof(image_buffer);
        for (size_t offset = 0; offset < frame_size;) {
            size_t to_write = offset + CHUNK_SIZE < frame_size ? CHUNK_SIZE : frame_size - offset;
            int written = usb_serial_jtag_write_bytes(image_buffer + offset, to_write, pdMS_TO_TICKS(1000));
            if (written < to_write) {
                vTaskDelay(1);
            }
            if (written > 0) {
                offset += written;
            }
        }
    }

    // Wait ~1 second
    vTaskDelay(pdMS_TO_TICKS(1000));
}

// ---------- ESP-IDF entry point ----------

extern "C" void app_main()
{
    setup();
    while (true) {
        loop();
    }
}
