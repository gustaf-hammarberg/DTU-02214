# Lab 1: Keyword Spotter for Seeed Studio XIAO ESP32-S3 Sense

A *keyword spotter* is an algorithm that continuously analyzes an audio stream from a microphone and recognizes certain words or speech commands in the audio, for example "start", "stop", "yes" or "no". Well-known examples of products with keyword spotters include Amazon's Alexa, which wakes up when it hears the keyword "Alexa", or Google applications that wake up when they hear "Hey Google" (such keyword spotters are also referred to as *wake word detectors*). Keyword spotting is also commonly used in industrial applications where a voice interface is more practical than buttons or touch screens, for example controlling a hoist mechanism with the keywords "up", "down" and "stop".

This lab demonstrates the entire workflow for building a keyword spotter that detects the words "yes" and "no" on a Seeed Studio XIAO ESP32-S3 Sense. The specification is simple: When it hears the word "yes", it lights up the LED on the board, and when it hears "no", it turns off the LED. All the code and training data are given, so the tasks that you need to complete are:
 * Install and configure the tools and libraries needed to run the training and build the microcontroller application.
 * Run training, build and deploy the application, and verify that it works on your board.
 * Answer the review questions at the end of this document.

For this lab, you will need:
 * A computer running Windows, Linux or MacOS.
 * A Seeed Studio XIAO ESP32-S3 Sense board including the camera+microphone extension board.
 * A USB-C cable to connect the board to your computer.

The lab is divided into two parts:
 * The Python part, which downloads and preprocesses the training data, trains the model, and generates C code that will be included in the microcontroller application.
 * The ESP32 part, where you build and flash the microcontroller application to your board.

The Python and ESP32 parts are further detailed in their own README files:
 * [Python](python/README.md)
 * [ESP32](esp32/README.md)

## Review questions

After verifying that the keyword spotter works on your board, look at the code and answer the following questions:

1. What's the dimensionality (the total number of features) of a training example input before and after preprocessing?
2. Although inference is primarily associated with the final application in C++, inference also takes place in the training Python code. Identify the Python lines in `main.py` that perform inference.
3. Name at least one reason for why the generated model file `python/gen/model.tflite` is so much smaller than `python/gen/model.keras`.
4. In the ESP32 project, identify the C++ code lines (file and line numbers) that perform the following parts of the inference pipeline:
    * Data acquisition
    * Preprocessing
    * Inference

## And last but not least

Don't be alarmed if you don't understand all the code in this project! We will go through all the ins and outs of the training and inference pipelines in the next few lectures.

Good luck!