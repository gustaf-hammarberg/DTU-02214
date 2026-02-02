#include "preprocess.h"
#include "signal/src/kiss_fft_wrappers/kiss_fft_float.h"
#include "esp_log.h"

static float feature_buffer[SPECTRUM_WIDTH * SPECTRUM_HEIGHT]; // Circular buffer for features
static float hamming_window[FRAME_SIZE];                  // Hamming window coefficients
static float windowed_frame[FRAME_SIZE];                  // Windowed audio frame (only used locally in preprocess_put_audio)
static size_t stride = 0;                                 // Sliding window stride in number of frames specified by caller
static size_t write_frame_index = 0;                      // Frame index to next write into feature_buffer
static size_t read_frame_index = 0;                       // Frame index to next read from feature_buffer
static kiss_fft_float::kiss_fftr_cfg kfft_cfg;            // KissFFT configuration
static float amplitude = 0.0;                             // Amplitude of audio frames since last feature extraction
static const char *TAG = "Preprocess";

/**
 * @brief Initialize preprocessing. Call this once before calling preprocess_put_audio() or
 * preprocess_get_features().
 * 
 * @param stride Sliding window stride in number of frames.
 * @return       True if initialization was successful, false otherwise.
 */
bool preprocess_init(size_t stride)
{
    // This code was taken from espressif__esp-tflite-micro/tensorflow/lite/experimental/microfrontend/lib/fft_util.cc.
    // TODO: Maybe use the microfrontend library

    // Validate stride
    if (stride == 0 || stride > SPECTRUM_WIDTH)
    {
        ESP_LOGE(TAG, "Invalid stride %u (expected 1..%u)", (unsigned)stride, (unsigned)SPECTRUM_WIDTH);
        return false;
    }

    // Copy stride
    ::stride = stride;

    // Ask kissfft how much memory it wants
    size_t scratch_size = 0;
    kfft_cfg = kiss_fft_float::kiss_fftr_alloc(FRAME_SIZE, 0, nullptr, &scratch_size);
    if (kfft_cfg != nullptr)
    {
        ESP_LOGE(TAG, "Kiss memory sizing failed.");
        return 0;
    }
    void *scratch = malloc(scratch_size);
    if (scratch == nullptr)
    {
        ESP_LOGE(TAG, "Failed to alloc fft scratch buffer");
        return 0;
    }

    // Let kissfft configure the scratch space we just allocated
    kfft_cfg = kiss_fft_float::kiss_fftr_alloc(FRAME_SIZE, 0, scratch, &scratch_size);
    if (kfft_cfg != scratch)
    {
        ESP_LOGE(TAG, "Kiss memory preallocation strategy failed.");
        return 0;
    }

    // Precompute Hamming window
    for (int i = 0; i < FRAME_SIZE; i++)
    {
        hamming_window[i] = 0.54 - 0.46 * cos(2 * M_PI * (float)i / (float)(FRAME_SIZE - 1));
    }

    return 1;
}

/**
 * @brief Put an audio frame into the preprocessing pipeline. Call this once per audio frame.
 * Afterwards, call preprocess_get_features() to check if a complete preprocessed feature window
 * is ready for inference.
 *
 * @param audio_frame Pointer to an array of FRAME_SIZE audio samples.
 */
void preprocess_put_audio(const float *audio_frame)
{
    // This function applies the same preprocessing as the following Python code:
    // frame = frame - np.average(frame)
    // frame = frame * np.hamming(FRAME_SIZE)
    // spectral_frame = np.abs(np.fft.rfft(frame))
    // spectral_frame = np.log1p(spectral_frame)
    // spectral_frame = (spectral_frame - SPECTRUM_MEAN) / SPECTRUM_STD

    // Compute mean
    float sum = 0.0, min = 32767.0f, max = -32768.0f;
    for (int i = 0; i < FRAME_SIZE; i++)
    {
        float value = audio_frame[i];
        sum += value;
        if (value > max)
            max = value;
        if (value < min)
            min = value;
    }
    float mean = sum / FRAME_SIZE;

    // Compute and remember amplitude
    float frame_amplitude = (max - min) / 2.0f;
    if (frame_amplitude > amplitude)
        amplitude = frame_amplitude;

    // Subtract mean and apply Hamming window
    for (int i = 0; i < FRAME_SIZE; i++)
    {
        windowed_frame[i] = (audio_frame[i] - mean) * hamming_window[i];
    }

    // Compute FFT using kiss_fftr
    static kiss_fft_float::kiss_fft_cpx spectrum[FRAME_SIZE / 2 + 1];
    kiss_fft_float::kiss_fftr(kfft_cfg, windowed_frame, spectrum);

    // Compute (log(1 + abs(spectrum)) - SPECRUM_MEAN) / SPECTRUM_STD
    float *write_pointer = feature_buffer + write_frame_index * SPECTRUM_HEIGHT;
    for (int i = 1; i < SPECTRUM_HEIGHT + 1; i++) // +1 to skip DC component
    {
        float real = spectrum[i].r;
        float imag = spectrum[i].i;
        float absolute = sqrt(real * real + imag * imag);
        float unnormalized = log1p(absolute);
        float normalized = (unnormalized - SPECTRUM_MEAN) / SPECTRUM_STD;
        *write_pointer++ = normalized;
    }

    // Advance write index
    write_frame_index = (write_frame_index + 1) % SPECTRUM_WIDTH;
}

/**
 * @brief Get features and amplitude. Call this once per audio frame. If it returns true, a full
 * window of features has been copied to the features array.
 *
 * @param features  Pointer to a array of SPECTRUM_WIDTH * SPECTRUM_HEIGHT floats.
 * @param amplitude Pointer to a float that will be filled with the amplitude of audio that
 *                  produced the feature window.
 * @return          True if features have been copied and are ready for inference, false otherwise.
 */
bool preprocess_get_features(float *features, float *amplitude)
{
    // Determine if a full feature window is ready
    if (read_frame_index != write_frame_index)
    {
        return false;
    }

    // Copy feature buffer to features
    size_t part1_float_count = (SPECTRUM_WIDTH - read_frame_index) * SPECTRUM_HEIGHT;
    size_t part2_float_count = read_frame_index * SPECTRUM_HEIGHT;
    memcpy(features, feature_buffer + read_frame_index * SPECTRUM_HEIGHT, part1_float_count * sizeof(float));
    if (part2_float_count > 0)
    {
        memcpy(features + part1_float_count, feature_buffer, part2_float_count * sizeof(float));
    }

    // Advance read index
    read_frame_index = (read_frame_index + stride) % SPECTRUM_WIDTH;

    // Copy amplitude
    *amplitude = ::amplitude;
    ::amplitude = 0.0;

    return true;
}
