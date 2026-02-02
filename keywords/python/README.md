# Lab 1, Part 1: Training a Keyword Spotter with Python and TensorFlow

This part of the lab runs the entire training pipeline for the Keyword Spotter, introduced in [../README.md](../README.md). As you can see at the bottom of `main.py`, it performs seven functions, namely:

1. *Download data*. This function downloads and extracts all the training data into the `../data` folder. The training data are regular `.wav` sound files sorted in folders corresponding to the classes "yes", "no" and "other". You are encouraged to open and listen to some of them.
2. *Preprocess data*. This function loads, processes, splits and converts the training data into consolidated 3D arrays in the format expected by TensorFlow's training functions. The arrays are saved as `.npy` files in the `gen` folder. More precisely, the preprocessing function includes the following substeps:
    * Load the raw audio data. Each sound file is loaded as a 1D array of length ~16000, corresponding to one second of audio sampled with 16 kHz.
    * Perform audio signal processing: Mean removal, Hamming window, Fourier transform, absolute value, logarithm, and normalization. Each sound file is now represented by a spectrogram as a 2D array of shape (62, 64): 62 spectral frames with 64 spectral bins each.
    * Arrange into a single 3D array of equal-length spectrograms. Since the total number of examples (sound files) is 3162, the shape is (3162, 62, 64).
    * Generate a 1D label array of length 3162, with class indices 0 ("other"), 1 ("yes"), 2 ("no"), or  matching the examples in the data array.
    * Shuffle the examples, so the three classes are well distributed over the example axis.
    * Split into training (~60% = 1897 examples), test (~20% = 632 examples) and validation (~20% = 633 examples) data sets, and save as `.npy` files.
3. *Train the model*. This function creates and initializes a neural network model, primarily based on 1D-convolution, and uses TensorFlow to train the model on the training set until the validation loss no longer improves. This strategy is called "early stopping" because training is set to run for 100 epochs, but training beyond the point where the model's loss on the validation set no longer improves is just a waste of time, and more importantly, it might result in an overfit model. The final trained model is saved as a `.keras` file in the `gen` folder. You can use the application [Netron](https://netron.app) to inspect the file.
4. *Evaluate the model*. This function calculates and prints performance metrics for the model, in particular test accuracy and confusion matrix, to the console.
5. *Convert and optimize the model*. This function converts the model to the TensorFlow Lite (TFLite) format, which is necessary for deployment on embedded devices. At the same time, the model is quantized from 32-bit floating point numbers to 8-bit integers. This makes the model much smaller and faster at the cost of slightly reduced accuracy. The converted and quantized model is saved as a `.tflite` file in the `gen` folder, and you can use the Netron to also inspect this file. More importantly, the model is also dumped as a byte array to a `.c` file and accompanying header (`.h`) file in the `../esp32/main` folder for inclusion in the microcontroller application build.
6. *Evaluate the converted model*. This function calculates and prints performance metrics for the model to the console.
7. *Generate a test case*. This function takes a randomly chosen sound file and the TFLite model to generate test vectors for use in inference pipeline tests in C. The purpose is to verify that the preprocessing, quantization and inference produce equivalent results in the C code as in Python. The test vectors are dumped in a `.h` file in the `../esp32/main` folder for inclusion in the microcontroller application, which will run the test on startup and abort if the test fails.

## 1. Install Python 3.13

Install Python 3.13:

 * On Windows, download the Python 3.13 installer from https://www.python.org/downloads/.
 * On Mac and Linux, open a terminal and enter:
   ```sh
   sudo apt install -y python3 python3-venv python3-pip python-is-python3
   ```

Note that TensorFlow, which is the machine learning library that we will use, is currently not compatible with Python 3.14, so make sure you install version 3.13. If you already have Python of any version between and including 3.10 and 3.13 installed on your computer, you can skip this step.

## 2. Create a virtual environment

Create a virtual environment for the course as follows:

1. Open a terminal and `cd` to any location where you want to place the virtual environment folder, for example the repository root. Note that you will reuse the same virtual environment for future labs and projects.
2. Create the virtual environment:
    ```sh
    python -m venv .venv
    ```

## 3. Install a Python IDE

Install an IDE for editing Python code, for example Visual Studio Code (VS Code) from https://code.visualstudio.com/download with the Python extension. You're welcome to use any other Python IDE that you prefer - another good option is [PyCharm](https://www.jetbrains.com/pycharm/).
 * *In VS Code, I recommend setting `python.analysis.typeCheckingMode` to `off` in your user settings to avoid false error squiggles in the editor, since the Python extension isn't capable of handling TensorFlow types. You can open the user settings by pressing `Ctrl+Shift+P` (Windows/Linux) or `Command+Shift+P` (Mac) and enter `Preferences: Open User Settings (JSON)`, then find and change the setting `"python.analysis.typeCheckingMode"` to `"off"`.*

## 4. Open and configure the Keyword Spotter Python project

1. Open the folder `labs/keywords/python` in the IDE.
2. Configure the project to use the virtual environment that you created above.
    * *In VS Code, you can click the Python environment indicator in the bottom right corner of the window, select `Browse...` and select the Python executable `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python3` (Mac/Linux) under the `.venv` folder of the virtual environment that you created in step 2.*
3. Open a terminal in the IDE.
4. Verify that the correct virtual environment is activated by checking that this command outputs the correct path to its `.venv` folder:
    ```sh
    python -c "import sys; print(sys.prefix)"
    ```
5. Install dependencies:
    ```sh
    python -m pip install -r requirements.txt
    ```

## 5. Run the training script

1. Make sure that your terminal is in the `labs/keywords/python` folder.
2. Run the training script:
    ```sh
    python main.py
    ```
3. Verify that the test accuracy after quantization (printed to the terminal) is at least 94%. If not, run the script again.
4. Verify that the following C source and header files have been generated:
    * `../esp32/main/model.c`
    * `../esp32/main/model.h`
    * `../esp32/main/test_case.h`
