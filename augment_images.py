import albumentations as A
import cv2
import os
import glob
import numpy as np

# 1. Augmentation Pipeline Configuration
# These transformations will be applied randomly to produce diverse images
transform = A.Compose([
    A.Rotate(limit=15, p=0.7),                # Rotate the image left/right by a maximum of 15 degrees
    A.RandomBrightnessContrast(p=0.5),        # Randomly change brightness and contrast
    A.RandomScale(scale_limit=0.1, p=0.5),    # Simple Zoom (Scaling)
    # A.HorizontalFlip(p=0.5)                 # ⚠️ Enable this only if left/right orientation doesn't matter
])

def augment_class(class_name, input_dir, output_dir, multiplier=4):
    """
    class_name: Name of the movement (used for naming the images).
    input_dir: Path to original images.
    output_dir: Path where new images will be saved.
    multiplier: Number of copies per image (9 copies + original = 10).
    """
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Fetch all images from the directory
    image_paths = glob.glob(os.path.join(input_dir, "*.*")) # Reads jpg, png, etc.
    print(f"📂 Found {len(image_paths)} images in class: {class_name}")

    count = 0
    for img_path in image_paths:
        # Read the image
        image = cv2.imread(img_path)
        if image is None: continue

        # 1. Save the original image first
        original_name = f"{class_name}_orig_{count}.jpg"
        cv2.imwrite(os.path.join(output_dir, original_name), image)
        count += 1

        # 2. Create augmented versions
        for i in range(multiplier):
            try:
                augmented = transform(image=image)['image']
                # Save the augmented image
                aug_name = f"{class_name}_aug_{count}_{i}.jpg"
                cv2.imwrite(os.path.join(output_dir, aug_name), augmented)
            except Exception as e:
                pass
                
    print(f"✅ Done! You now have approximately {count * (multiplier+1)} images in {output_dir}")

# --- Usage ---
# Change paths according to your local directories

# Example:
augment_class("Clasped Hand", "C:/Users/anesr/OneDrive/Documents/Hand_Model_imges/Clasped Hand", "dataset_final/Clasped Hand")
# augment_class("Raised", "path/to/original_raised", "dataset_final/Raised")


# input_folder = "C:/Users/anesr/OneDrive/Documents/Hand_Model_imges/Closed Hand"  # مكان الصور الأصلية
# output_folder = "dataset/augmented_images" # مكان الحفظ الجديد
# classes = ["Open_Hand", "Closed_Hand", "Pointing"]
# images_per_original = 5  # كم نسخة نصنع من كل صورة؟
