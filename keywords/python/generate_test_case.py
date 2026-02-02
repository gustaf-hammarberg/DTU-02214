import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from preprocess import preprocess_audio, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT, NUM_CLASSES

TEST_AUDIO_FILE = '../data/other/audio_noise_1354.wav'  # Change this to the desired recording file


def generate_test_case(test_case_h_path: str):
    """
    Loads an audio file from the Data directory, cuts out a second of audio, and saves it as a test case in both JSON and C++.
    """
    # Load the audio file
    sample_rate, audio_data = wavfile.read(TEST_AUDIO_FILE)

    # Preprocess the same wave file to get the preprocessed window
    x_test = preprocess_audio(audio_data)
    if x_test.shape[0] != SPECTROGRAM_WIDTH or x_test.shape[1] != SPECTROGRAM_HEIGHT:
        raise ValueError(f'Expected preprocessed data shape ({SPECTROGRAM_WIDTH}, {SPECTROGRAM_HEIGHT}), but got {x_test.shape}')

    # Load interpreter and get quantization parameters
    interpreter = tf.lite.Interpreter(model_path='gen/model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    # Quantize it
    x_test_quantized = x_test / input_scale + input_zero_point
    x_test_quantized = np.clip(x_test_quantized, -128, 127)
    x_test_quantized_int = x_test_quantized.astype(np.int8)

    # Predict with the quantized input
    interpreter.set_tensor(input_details[0]['index'], x_test_quantized_int.reshape(1, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT))
    interpreter.invoke()
    y_pred_quantized = interpreter.get_tensor(output_details[0]['index'])[0]

    # Dequantize output
    y_pred = (y_pred_quantized.astype(np.float32) - output_zero_point) * output_scale

    # Save the test case in C++ format
    with open(test_case_h_path, 'w') as cpp_file:
        cpp_file.write('#ifndef TEST_CASE_H\n')
        cpp_file.write('#define TEST_CASE_H\n\n')

        cpp_file.write('#include <stdint.h>\n\n')

        cpp_file.write('#define TEST_LENGTH {}\n'.format(len(audio_data)))

        cpp_file.write('const int32_t raw_audio[TEST_LENGTH] = {\n')
        cpp_file.write(', '.join(map(str, audio_data)))
        cpp_file.write('\n};\n\n')

        cpp_file.write('const float test_x[{}] = {{\n'.format(SPECTROGRAM_WIDTH * SPECTROGRAM_HEIGHT))
        for row in x_test:
            cpp_file.write('    ' + ', '.join(map(str, row)) + ',\n')
        cpp_file.write('};\n\n')

        cpp_file.write('const int8_t test_xq[{}] = {{\n'.format(SPECTROGRAM_WIDTH * SPECTROGRAM_HEIGHT))
        for row in x_test_quantized_int:
            cpp_file.write('    ' + ', '.join(map(str, row)) + ',\n')
        cpp_file.write('};\n\n')

        cpp_file.write('const float test_prediction[{}] = {{ {} }};\n\n'.format(NUM_CLASSES, ', '.join(map(str, y_pred))))

        cpp_file.write('#endif // TEST_CASE_H\n')


if __name__ == '__main__':
    # Generate test case
    generate_test_case()
