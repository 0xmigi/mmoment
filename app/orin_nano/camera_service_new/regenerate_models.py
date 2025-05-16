#!/usr/bin/env python3
"""
Regenerate FaceNet Models

This script creates new TFLite and Keras models for the FaceNet implementation.
These are simple models that generate random embeddings but can be used to test the system.
"""

import os
import numpy as np
import tensorflow as tf
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ModelRegenerator")

def create_simple_model():
    """Create a simple model that mimics FaceNet's input/output structure"""
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation=None)(x)
    
    # Add L2 normalization layer
    outputs = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        name='l2_normalization'
    )(x)
    
    model = tf.keras.Model(inputs, outputs, name="facenet_model")
    return model

def convert_to_tflite(model, output_path):
    """Convert Keras model to TFLite format"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    logger.info(f"TFLite model saved to {output_path}")
    return True

def test_tflite_model(model_path):
    """Test that the TFLite model works properly"""
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create a dummy input
    dummy_input = np.zeros((1, 160, 160, 3), dtype=np.float32)
    
    # Run a test inference
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    logger.info(f"TFLite output shape: {output.shape}")
    logger.info(f"TFLite output norm: {np.linalg.norm(output)}")
    
    return True

def test_keras_model(model):
    """Test that the Keras model works properly"""
    # Create a dummy input
    dummy_input = np.zeros((1, 160, 160, 3), dtype=np.float32)
    
    # Run a test inference
    output = model.predict(dummy_input, verbose=0)
    
    logger.info(f"Keras output shape: {output.shape}")
    logger.info(f"Keras output norm: {np.linalg.norm(output)}")
    
    return True

def main():
    # Set up model directory
    model_dir = os.path.expanduser("~/mmoment/app/orin_nano/camera_service_new/models/facenet_model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up model paths
    keras_path = os.path.join(model_dir, "facenet_keras.h5")
    tflite_path = os.path.join(model_dir, "facenet_model.tflite")
    
    # Create model
    logger.info("Creating FaceNet-like model...")
    model = create_simple_model()
    
    # Test Keras model
    logger.info("Testing Keras model...")
    test_keras_model(model)
    
    # Save Keras model
    logger.info(f"Saving Keras model to {keras_path}...")
    model.save(keras_path)
    logger.info("Keras model saved successfully")
    
    # Convert to TFLite
    logger.info(f"Converting to TFLite and saving to {tflite_path}...")
    if convert_to_tflite(model, tflite_path):
        # Test TFLite model
        logger.info("Testing TFLite model...")
        test_tflite_model(tflite_path)
    
    logger.info("Model regeneration complete!")

if __name__ == "__main__":
    main() 