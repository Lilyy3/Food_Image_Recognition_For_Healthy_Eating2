import unittest
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model
# Import the image processing function we created for error handling
from image_utils import process_image  

class TestModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Load the model once before running any tests.
        Uses try/except to handle environments (like CI/CD) where the model file 
        might be missing (e.g., due to Git LFS limitations).
        """
        try:
            cls.model = load_model('efficientnetb0_food101_over75_best.keras')
        except Exception as e:
            print(f" Model not loaded (maybe missing in CI): {e}")
            cls.model = None
    
    def test_model_input_shape(self):
        """Test that the model expects the correct input shape (224x224x3)."""
        if self.model is None:
            self.skipTest("Model not available in this environment")
        self.assertEqual(self.model.input_shape, (None, 224, 224, 3))
    
    def test_prediction_output_shape(self):
        """Test that the model outputs predictions for exactly 101 food classes."""
        if self.model is None:
            self.skipTest("Model not available in this environment")
        dummy_input = np.random.rand(1, 224, 224, 3)
        pred = self.model.predict(dummy_input)
        self.assertEqual(pred.shape[1], 101)  # Food101 has 101 classes

    def test_preprocessing_failure_file_not_found(self):
        """Test Error Handling: Ensure FileNotFoundError is raised for missing files."""
        # Now the function checks extension first, so we need a valid extension for this test
        with self.assertRaises(FileNotFoundError):
            process_image("nonexistent.jpg")  # Use .jpg extension so it passes extension check

    def test_preprocessing_invalid_extension(self):
        """Test Error Handling: Ensure ValueError is raised for non-image file types."""
        # This will now raise ValueError (extension check first)
        with self.assertRaises(ValueError):
            process_image("fake_file.txt")
    
    def test_preprocessing_corrupted_image(self):
        """Test Error Handling: Ensure ValueError is raised for corrupted/empty images."""
        # Create an invalid (empty/corrupted) file to simulate a bad image
        with open('corrupted.jpg', 'w') as f:
            f.write('this is not an image')
        
        # Verify that our function catches the corruption and raises an error
        with self.assertRaises(ValueError):
            process_image('corrupted.jpg')
        
        # Clean up the temporary file after the test
        if os.path.exists('corrupted.jpg'):
            os.remove('corrupted.jpg')

if __name__ == '__main__':
    unittest.main()
