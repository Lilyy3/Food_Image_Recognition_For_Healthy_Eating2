import os
from PIL import Image

def process_image(image_path):
    """
    Safely process an image file with comprehensive error handling.
    
    Validates:
    1. File existence.
    2. File extension (only JPG, JPEG, PNG).
    3. Image integrity (not corrupted or empty).
    
    Args:
        image_path (str): Path to the input image file.
        
    Returns:
        PIL.Image: The processed image (RGB, resized to 224x224).
        
    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file extension is invalid, or if the image is corrupted/empty.
    """
    # 1. Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    
    # 2. Validate the file extension (allow only specific image types)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    if not image_path.lower().endswith(valid_extensions):
        raise ValueError(f"Unsupported file type. Allowed types: {', '.join(valid_extensions)}")
    
    # 3. Attempt to open the image and validate its content
    try:
        with Image.open(image_path) as img:
            # Check if the image has zero dimensions (empty/fake)
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError("Image is empty or corrupted (zero dimensions).")
            
            # Convert to RGB format (required by most models)
            img = img.convert('RGB')
            # Resize to the target dimensions expected by EfficientNetB0
            img = img.resize((224, 224))
            return img
    
    except Exception as e:
        # Catch any other errors (e.g., unreadable files, corrupt JPEG headers)
        raise ValueError(f"Image is corrupted or invalid: {str(e)}")

# Example usage for local testing
if __name__ == "__main__":
    try:
        # This will raise an error because it's a Python file, not an image
        process_image("app_streamlit.py")  
    except Exception as e:
        print("Error caught successfully:", e)
