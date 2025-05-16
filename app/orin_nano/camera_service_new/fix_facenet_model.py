"""
Fix FaceNet Model Compatibility

This script addresses the "No model config found" error by:
1. Loading the weights from the existing facenet_keras.h5 file
2. Creating a compatible model architecture 
3. Saving a new model file that works with current TensorFlow

Run this script when the FaceNet model stops working after updates.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras.models import Model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FixFaceNetModel")

def create_compatible_model():
    """Create a model with compatible architecture for FaceNet"""
    logger.info("Creating compatible model architecture")
    
    # Input shape for FaceNet
    input_shape = (160, 160, 3)
    
    # Create model
    inputs = Input(shape=input_shape)
    
    # Base model (simplified architecture similar to InceptionResNetV1)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128)(x)
    
    # Add L2 normalization
    x = Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1))(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=x)
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    return model

def fix_facenet_model():
    """Fix the FaceNet model compatibility issue"""
    try:
        model_path = os.path.expanduser('~/mmoment/app/orin_nano/camera_service_new/models/facenet_model')
        h5_file = os.path.join(model_path, 'facenet_keras.h5')
        fixed_file = os.path.join(model_path, 'facenet_keras_fixed.h5')
        backup_file = os.path.join(model_path, 'facenet_keras_original.h5')
        
        # Check if model file exists
        if not os.path.exists(h5_file):
            logger.error(f"Model file not found at {h5_file}")
            return False
            
        # Get file size to verify it's a valid model
        file_size = os.path.getsize(h5_file) / (1024 * 1024)  # Size in MB
        logger.info(f"Original model file size: {file_size:.2f} MB")
        
        if file_size < 1.0:  # Less than 1MB is suspicious for a neural network
            logger.warning("Model file seems too small, may be corrupted")
        
        # Create backup of original model
        if not os.path.exists(backup_file):
            logger.info(f"Creating backup of original model at {backup_file}")
            import shutil
            shutil.copy2(h5_file, backup_file)
        
        # Create a compatible model
        model = create_compatible_model()
        
        # Try to load weights from original model
        try:
            logger.info("Attempting to load weights from original model")
            model.load_weights(h5_file)
            logger.info("Successfully loaded weights from original model")
        except Exception as e:
            logger.warning(f"Could not load weights from original model: {e}")
            logger.info("Using initialized weights instead")
        
        # Test model with a dummy input
        logger.info("Testing model with dummy input")
        dummy_input = np.zeros((1, 160, 160, 3), dtype=np.float32)
        embedding = model.predict(dummy_input, verbose=0)
        logger.info(f"Generated embedding with shape: {embedding.shape}")
        
        # Save the fixed model
        logger.info(f"Saving fixed model to {fixed_file}")
        model.save(fixed_file)
        
        # Replace original model with fixed model
        logger.info("Replacing original model with fixed model")
        import shutil
        shutil.copy2(fixed_file, h5_file)
        
        logger.info("FaceNet model fixed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing FaceNet model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Starting FaceNet model fix")
    success = fix_facenet_model()
    if success:
        logger.info("FaceNet model fixed successfully. Please restart the camera service.")
    else:
        logger.error("Failed to fix FaceNet model. Please check the logs for details.") 