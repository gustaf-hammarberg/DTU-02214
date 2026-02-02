#pragma once
    
bool camera_init(void);
bool camera_capture_frame(uint8_t *frame_buffer);

#define FRAME_W 320
#define FRAME_H 240
#define FRAME_C 2