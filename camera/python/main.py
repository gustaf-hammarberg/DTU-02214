import argparse
import os
import pygame
import sys
import serial
import time
from datetime import datetime

# Configuration constants
DEFAULT_OUTPUT_PATH = "../data/"        # Path to save captured images
DEFAULT_PORT = "COM3"                   # Default serial port; can be overridden with CLI argument --port
BAUD_RATE = 921600                      # Will be ignored, but we need to provide a value
SERIAL_TIMEOUT = 2.0                    # Serial read timeout in seconds
WIDTH = 320                             # Frame width in pixels - must match FRAME_W in firmware (camera.h)
HEIGHT = 240                            # Frame height in pixels - must match FRAME_H in firmware (camera.h)
FRAME_PREAMBLE = b"===FRAME===\n"       # Preamble sequence indicating start of frame - must match suffix of FRAME_PREAMBLE in firmware (main.cpp)


def capture_and_display_loop(port: str, output_path: str):
    # Open serial port
    print(f"Opening serial port {port}... ", end="")
    try:
        serial_port = serial.Serial(port, BAUD_RATE, timeout=2.0)
        serial_port.reset_input_buffer()
    except serial.SerialException as exc:
        print(f"Failed to open serial port {port}: {exc}", file=sys.stderr)
        return

    # Initialize pygame and open window
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Serial Camera Viewer")

    # Print instructions
    print("Connection established.")
    print("Press number keys (0-9) to save the current frame to corresponding class folder.")
    print("Press 'q' or 'ESC' to quit.")

    # Sens 'S' on serial port to start streaming
    serial_port.write(b'S')

    # Main loop
    last_surface = None
    running = True
    try:
        while running:
            # Handle input events and user actions
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif pygame.K_0 <= event.key <= pygame.K_9 and last_surface is not None:
                        class_index = event.key - pygame.K_0
                        _save_frame(output_path, last_surface, class_index)

            # Capture frame from serial port
            surface = _capture_frame(serial_port)
            if surface is None:
                continue

            # Remember last surface for saving
            last_surface = surface.copy()

            # Blit and present the frame
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            time.sleep(0.001)
    finally:
        # Release serial port and pygame resources
        serial_port.close()
        pygame.quit()


def _capture_frame(serial_port: serial.Serial) -> pygame.Surface | None:
    # Wait for preamble
    chunk = serial_port.read_until(FRAME_PREAMBLE)
    if not chunk.endswith(FRAME_PREAMBLE):
        print("Preamble timeout, retrying...")
        return None

    # Read a full frame after the preamble
    frame_rgb565 = serial_port.read(WIDTH * HEIGHT * 2)
    if len(frame_rgb565) != WIDTH * HEIGHT * 2:
        print(f"Incomplete frame received ({len(frame_rgb565)} bytes), skipping...")
        return None

    # Convert from RGB565 to RGB888
    frame_rgb = bytearray(WIDTH * HEIGHT * 3)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            src_index = (y * WIDTH + x) * 2
            dst_index = (y * WIDTH + x) * 3
            byte1 = frame_rgb565[src_index]
            byte2 = frame_rgb565[src_index + 1]
            r8 = byte1 & 0xF8
            g8 = ((byte1 & 0x07) << 5) | ((byte2 & 0xE0) >> 3)
            b8 = (byte2 & 0x1F) << 3
            frame_rgb[dst_index] = r8
            frame_rgb[dst_index + 1] = g8
            frame_rgb[dst_index + 2] = b8

    # Create and return pygame surface
    return pygame.image.frombuffer(frame_rgb, (WIDTH, HEIGHT), "RGB")


def _save_frame(output_path: str, surface: pygame.Surface, class_index: int):
    # Create directory if needed
    directory = os.path.join(output_path, str(class_index))
    os.makedirs(directory, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.png"
    path = os.path.join(directory, filename)

    # Do save image
    pygame.image.save(surface, path)
    print(f"Saved image to {path}.")


if __name__ == "__main__":
    # Define CLI arguments
    parser = argparse.ArgumentParser(description="Serial RGB frame viewer")
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"Serial port (default: {DEFAULT_PORT})")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help=f"Output directory (default: {DEFAULT_OUTPUT_PATH})")

    # Parse arguments
    args = parser.parse_args()

    # Run capture and display loop
    capture_and_display_loop(args.port, args.output_path)
