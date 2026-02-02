#include "esp_log.h"
#include "model.h"

// Include TFLM
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// Tensor arena size, found by trial and error
#define TENSOR_ARENA_SIZE (30 * 1024)

// Static variables
static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;
static const char *TAG_INF = "Inference";

/**
 * @brief Initialize the TFLite Micro interpreter with the model.
 * 
 * @return True if initialization was successful, false otherwise.
 */
bool inference_init()
{
    // Load TFlite model
    model = tflite::GetModel(model_binary);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG_INF, "Model schema mismatch!");
        return false;
    }

    // Create an interpreter
    static tflite::MicroMutableOpResolver<9> micro_op_resolver;
    // Conv1D
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddConv2D();
    // MaxPool1D
    micro_op_resolver.AddMaxPool2D();
    // Flatten
    micro_op_resolver.AddShape();
    micro_op_resolver.AddExpandDims();
    micro_op_resolver.AddStridedSlice();
    micro_op_resolver.AddPack();
    // Dense with sigmoid activation
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    // Allocate memory for input and output tensors
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        ESP_LOGE(TAG_INF, "Failed to allocate tensors!");
        return false;
    }

    // Get pointers for input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Print input and output tensor types and dimensions
    ESP_LOGI(TAG_INF, "Input tensor type: %s, shape: %d, %d, %d",
             TfLiteTypeGetName(input->type), input->dims->data[0], input->dims->data[1], input->dims->data[2]);
    ESP_LOGI(TAG_INF, "Output tensor type: %s, shape: %d, %d",
             TfLiteTypeGetName(output->type), output->dims->data[0], output->dims->data[1]);
    return true;
}

/**
 *  @brief Quantize the feature matrix from float to int8 and push it into the interpreter.
 *  @param features Pointer to the input feature matrix in float format.
 *  @return Pointer to the output feature matrix in int8 format.
 */
int8_t* inference_put_features(const float *features)
{
    // Quantize the feature matrix from float to int8 into the interpreter's input tensor
    for (size_t i = 0; i < SPECTRUM_WIDTH * SPECTRUM_HEIGHT; ++i)
    {
        float val_quant_float = roundf(features[i] / input->params.scale) + input->params.zero_point;
        if (val_quant_float > 127.0f)
        {
            val_quant_float = 127.0f;
        }
        else if (val_quant_float < -128.0f)
        {
            val_quant_float = -128.0f;
        }
        input->data.int8[i] = static_cast<int8_t>(val_quant_float);
    }

    // Return pointer to the quantized input matrix (used in test function)
    return input->data.int8;
}

/**
 * @brief Run inference on the model, obtain the prediction and dequantize to float.
 * @param prediction Pointer to store the prediction result. Expected to be of size NUM_CLASSES.
 * @return True if inference was successful, false otherwise.
 */
bool inference_predict(float *prediction)
{
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk)
    {
        return false;
    }

    // Dequantize the output from int8 to float
    for (size_t i = 0; i < NUM_CLASSES; ++i)
    {
        prediction[i] = (static_cast<float>(output->data.int8[i]) - output->params.zero_point) * output->params.scale;
    }

    return true;
}

