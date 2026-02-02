import os
import shutil
import ssl
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from generate_test_case import generate_test_case
from preprocess import *
from utils.export_tflite import write_model_h_file, write_model_c_file
from utils.eval_utils import compute_precision_recall_f1, print_confusion_matrix

# Minimize TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

DATA_DIR = '../data/'
GEN_DIR = 'gen/'
MODEL_C_PATH = '../esp32/main/model.c'
MODEL_H_PATH = '../esp32/main/model.h'
TEST_CASE_H_PATH = '../esp32/main/test_case.h'
USE_CACHED_DATA = True  # Set to True to reuse cached preprocessed data, False to force preprocess data


def download_data():
    # Download data if not present
    if not os.path.exists(os.path.join(DATA_DIR, 'README.md')):
        os.makedirs(DATA_DIR, exist_ok=True)

        # Bypass SSL verification for development
        ssl._create_default_https_context = ssl._create_unverified_context

        # Do download
        keras.utils.get_file(
            'yes_no_other.zip',
            origin='https://courses.compute.dtu.dk/02214/data/yes_no_other.zip',
            extract=True,
            cache_dir=DATA_DIR,
            cache_subdir='.cache')

        # Recursively move extracted files to DATA_DIR
        extracted_dir = os.path.join(DATA_DIR, '.cache', 'yes_no_other_extracted')
        for root, _, files in os.walk(extracted_dir):
            for file in files:
                src_path = os.path.join(root, file)
                relative_path = os.path.relpath(src_path, extracted_dir)
                dest_path = os.path.join(DATA_DIR, relative_path)
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                os.rename(src_path, dest_path)

        # Remove extracted directory
        shutil.rmtree(os.path.join(DATA_DIR, '.cache', 'yes_no_other_extracted'))


def preprocess_and_load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Preprocess data if not done already
    if not USE_CACHED_DATA \
            or not os.path.exists(GEN_DIR + 'x_train.npy') or not os.path.exists(GEN_DIR + 'y_train.npy') \
            or not os.path.exists(GEN_DIR + 'x_val.npy') or not os.path.exists(GEN_DIR + 'y_val.npy') \
            or not os.path.exists(GEN_DIR + 'x_test.npy') or not os.path.exists(GEN_DIR + 'y_test.npy'):
        preprocess_all(DATA_DIR, GEN_DIR)

    # Load preprocessed data
    x_train = np.load(GEN_DIR + 'x_train.npy')
    y_train = np.load(GEN_DIR + 'y_train.npy')
    x_val = np.load(GEN_DIR + 'x_val.npy')
    y_val = np.load(GEN_DIR + 'y_val.npy')
    x_test = np.load(GEN_DIR + 'x_test.npy')
    y_test = np.load(GEN_DIR + 'y_test.npy')

    return x_train, y_train, x_val, y_val, x_test, y_test


def train_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> keras.models.Model:
    # Build and compile model
    print('Building model...')
    model = Sequential([
        Conv1D(16, 5, activation='relu', input_shape=(SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT)),  # Output shape (58, 16)
        MaxPooling1D(2),  # Output shape (29, 16)
        Dropout(0.1),
        Conv1D(32, 5, activation='relu'),  # Output shape (25, 32)
        MaxPooling1D(2),  # Output shape (12, 32)
        Dropout(0.1),
        Conv1D(32, 3, activation='relu'),  # Output shape (10, 32)
        MaxPooling1D(2),  # Output shape (5, 32)
        Dropout(0.1),
        Flatten(),  # Output shape (160)
        Dense(NUM_CLASSES, activation='softmax')  # Output shape (3)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Train model with early stopping; save best model
    print('Training model...')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=16)
    model_checkpoint = keras.callbacks.ModelCheckpoint(GEN_DIR + 'model.keras', monitor='val_loss', save_best_only=True)
    model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val),
              callbacks=[early_stopping, model_checkpoint])

    # Load and return best model
    model = keras.models.load_model(GEN_DIR + 'model.keras')
    return model


def evaluate_model(model: keras.models.Model, x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    # Evaluate model on validation and test sets
    val_loss, val_accuracy = model.evaluate(x_val, y_val)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    y_pred_int = np.argmax(y_pred, axis=1)
    precision_yes, recall_yes, _ = compute_precision_recall_f1(y_test, y_pred_int, class_index=1)
    precision_no, recall_no, _ = compute_precision_recall_f1(y_test, y_pred_int, class_index=2)

    # Print evaluation metrics
    print()
    print('Validation loss:     %.4f' % val_loss)
    print('Validation accuracy: %.4f' % val_accuracy)
    print('Test loss:           %.4f' % test_loss)
    print('Test accuracy:       %.4f' % test_accuracy)
    print('Precision (yes):     %.4f' % precision_yes)
    print('Recall (yes):        %.4f' % recall_yes)
    print('Precision (no):      %.4f' % precision_no)
    print('Recall (no):         %.4f' % recall_no)
    print()

    # Print confusion matrix
    print_confusion_matrix(y_test, y_pred_int, ['other', 'yes', 'no'])


def export_model_to_tflite(model: keras.models.Model, x_train: np.ndarray, enable_quantization: bool = True) -> object:
    # Set up TensorFlow Lite converter
    print('Converting to TensorFlow Lite model...')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if enable_quantization:
        # Function for generating representative data
        def representative_dataset():
            yield [x_train.astype(np.float32)]

        # Set up quantization parameters
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    # Convert to TensorFlow Lite model
    tflite_model = converter.convert()

    # Print quantization scale and zero point
    if enable_quantization:
        # Load model in interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Do print
        print('Input scale:', input_details[0]['quantization'][0])
        print('Input zero point:', input_details[0]['quantization'][1])
        print('Output scale:', output_details[0]['quantization'][0])
        print('Output zero point:', output_details[0]['quantization'][1])

    # Export TensorFlow Lite model to C source files
    print('Exporting TensorFlow Lite model to C source files...')
    defines = {
        'SAMPLE_RATE': SAMPLE_RATE,
        'NUM_CLASSES': NUM_CLASSES,
        'FRAME_SIZE': FRAME_SIZE,
        'FRAME_STRIDE': FRAME_STRIDE,
        'SPECTRUM_WIDTH': SPECTROGRAM_WIDTH,
        'SPECTRUM_HEIGHT': SPECTROGRAM_HEIGHT,
        'SPECTRUM_MEAN': f'{SPECTRUM_MEAN}f',
        'SPECTRUM_STD': f'{SPECTRUM_STD}f',
    }
    declarations = []
    write_model_h_file(MODEL_H_PATH, defines, declarations)
    write_model_c_file(MODEL_C_PATH, tflite_model)

    # Save TensorFlow Lite model
    with open(GEN_DIR + 'model.tflite', 'wb') as f:
        f.write(tflite_model)

    return tflite_model


def evaluate_tflite_model(tflite_model: object, x_test: np.ndarray, y_test: np.ndarray):
    # Load interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    # Quantize x_test
    x_test_quantized = x_test / input_scale + input_zero_point
    x_test_quantized = np.clip(x_test_quantized, -128, 127)
    x_test_quantized_int = x_test_quantized.astype(np.int8)

    # Predict
    y_pred_quantized = np.empty((len(x_test_quantized_int), NUM_CLASSES), dtype=np.int8)
    for i in range(len(x_test_quantized_int)):
        interpreter.set_tensor(
            input_details[0]['index'], x_test_quantized_int[i].reshape(1, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT)
        )
        interpreter.invoke()
        y_pred_quantized[i] = interpreter.get_tensor(output_details[0]['index'])[0]

    # Dequantize output
    output_f32 = np.round((y_pred_quantized.astype(np.float32) - output_zero_point) * output_scale)

    # Compute evaluation metrics
    correct_predictions = 0
    for i in range(len(y_test)):
        if np.argmax(output_f32[i]) == y_test[i]:
            correct_predictions += 1
    test_accuracy = correct_predictions / len(y_test)
    y_pred_int = np.argmax(output_f32, axis=1)
    precision_yes, recall_yes, _ = compute_precision_recall_f1(y_test, y_pred_int, class_index=1)
    precision_no, recall_no, _ = compute_precision_recall_f1(y_test, y_pred_int, class_index=2)

    # Print evaluation metrics
    print()
    print('Test accuracy:       %.4f' % test_accuracy)
    print('Precision (yes):     %.4f' % precision_yes)
    print('Recall (yes):        %.4f' % recall_yes)
    print('Precision (no):      %.4f' % precision_no)
    print('Recall (no):         %.4f' % recall_no)
    print()

    # Print confusion matrix
    print_confusion_matrix(y_test, y_pred_int, ['other', 'yes', 'no'])


if __name__ == '__main__':
    # Download data set
    download_data()

    # Preprocess and load data
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_and_load_data()

    # Train model
    model = train_model(x_train, y_train, x_val, y_val)

    # Evaluate model
    evaluate_model(model, x_val, y_val, x_test, y_test)

    # Save TFLite model
    tflite_model = export_model_to_tflite(model, x_train)

    # Evaluate TFLite model
    evaluate_tflite_model(tflite_model, x_test, y_test)

    # Generate test case
    generate_test_case(TEST_CASE_H_PATH)

    print('Done.')
