import itertools
import os
import shutil

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras_tuner import BayesianOptimization
from keras_tuner.engine.trial import Trial

from main import download_data, preprocess_and_load_data
from preprocess import SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT, NUM_CLASSES


# Function to train a model with given hyperparameters
def _test_hyperparameters(hyperparameters: dict, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> float:
    first_layer = [
        Input(shape=(SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT)),
        Conv1D(hyperparameters['first_layer_size'], hyperparameters['kernel_size'], activation='relu'),
        MaxPooling1D(2),
    ]
    middle_layers = []
    for _ in range(hyperparameters['num_layers'] - 2):
        middle_layers.extend([
            Conv1D(hyperparameters['layer_size'], hyperparameters['kernel_size'], activation='relu'),
            MaxPooling1D(2),
            Dropout(hyperparameters['dropout_rate'])
        ])
    last_layers = [
        Conv1D(hyperparameters['layer_size'], hyperparameters['last_kernel_size'], activation='relu'),
        MaxPooling1D(2),
        Dropout(hyperparameters['dropout_rate']),
        Flatten(),
        Dense(NUM_CLASSES, activation='softmax')
    ]
    all_layers = [*first_layer, *middle_layers, *last_layers]

    # Check that the kernel sizes and number of layers do not reduce the output to less than 1 in the Conv1D layers
    output_length = SPECTROGRAM_WIDTH
    for layer in all_layers:
        if isinstance(layer, Conv1D):
            output_length = output_length - (layer.kernel_size[0] - 1)
        elif isinstance(layer, MaxPooling1D):
            output_length = output_length // layer.pool_size[0]
        if output_length < 1:
            print('Invalid', end=' ')
            return 0.0

    # Build and compile model
    model = Sequential(all_layers)
    model.compile(optimizer=Adam(learning_rate=hyperparameters['learning_rate']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model with early stopping
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ]
    model.fit(x_train, y_train, epochs=100, batch_size=hyperparameters['batch_size'], validation_data=(x_val, y_val), callbacks=callbacks, verbose=0)

    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)

    return val_accuracy


def tune_hyperparameters_bayesian(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> dict:
    # Subclass BayesianOptimization to define the search space and evaluation function
    class MyTuner(BayesianOptimization):
        def run_trial(self, trial: Trial, x_train, y_train, x_val, y_val):
            hp = trial.hyperparameters
            hyperparameters = {
                'learning_rate': hp.Choice('learning_rate', [0.005, 0.001, 0.0005]),
                'batch_size': hp.Choice('batch_size', [32, 64, 128]),
                'dropout_rate': hp.Choice('dropout_rate', [0.0, 0.1, 0.2]),
                'num_layers': hp.Choice('num_layers', [2, 3, 4]),
                'layer_size': hp.Choice('layer_size', [16, 32, 64]),
                'first_layer_size': hp.Choice('first_layer_size', [16, 32, 64]),
                'kernel_size': hp.Choice('kernel_size', [3, 5]),
                'last_kernel_size': hp.Choice('last_kernel_size', [3, 5])
            }
            return _test_hyperparameters(hyperparameters, x_train, y_train, x_val, y_val)

    # Run tuning
    tuner = MyTuner(objective='val_accuracy', max_trials=100, directory='gen', project_name='tuning')
    tuner.search(x_train, y_train, x_val, y_val)

    # Delete 'gen/tuning' directory, so we can run again
    shutil.rmtree("gen/tuning", ignore_errors=True)

    return tuner.get_best_hyperparameters(1)[0].values


def tune_hyperparameters_grid_search(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> dict:
    # Define hyperparameter search space
    hyperparameter_space = {
        'learning_rate': [0.001, 0.0005],
        'batch_size': [32, 64],
        'dropout_rate': [0.1, 0.2],
        'num_layers': [2, 3, 4],
        'layer_size': [16, 32],
        'first_layer_size': [16, 32],
        'kernel_size': [3, 5],
        'last_kernel_size': [3, 5]
    }

    # Iterate over all combinations of hyperparameters
    best_hyperparameters = None
    best_val_accuracy = 0
    for hyperparameter_values in itertools.product(*hyperparameter_space.values()):
        hyperparameters = dict(zip(hyperparameter_space.keys(), hyperparameter_values))
        print(hyperparameters, end='... ')

        # Train 5 times with the same hyperparameters to reduce variance
        accuracies = []
        for _ in range(5):
            val_accuracy = _test_hyperparameters(hyperparameters, x_train, y_train, x_val, y_val)
            accuracies.append(val_accuracy)

        # Average validation accuracy over 5 runs
        val_accuracy = sum(accuracies) / len(accuracies)
        print('Validation accuracy: %.4f' % val_accuracy)

        # Update best hyperparameters
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_hyperparameters = hyperparameters

    return best_hyperparameters


if __name__ == '__main__':
    # Download data set
    download_data()

    # Preprocess and load data
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_and_load_data()

    # Tune hyperparameters
    # best_hyperparameters = tune_hyperparameters_grid_search(x_train, y_train, x_val, y_val)
    best_hyperparameters = tune_hyperparameters_bayesian(x_train, y_train, x_val, y_val)

    print('Best hyperparameters:', best_hyperparameters)

