import os
import shutil
from pathlib import Path

# Paths
dataset_base = r"U:\Fraunhofer Waldbrand\datasets\SAINetset_v8.0\data"
train_images_dir = os.path.join(dataset_base, "train", "images")
train_labels_dir = os.path.join(dataset_base, "train", "labels")
val_images_dir = os.path.join(dataset_base, "val", "images")
val_labels_dir = os.path.join(dataset_base, "val", "labels")

# Output directory
smoke_output_dir = r"U:\Fraunhofer Waldbrand\smoke_images"
os.makedirs(smoke_output_dir, exist_ok=True)

def get_image_extension(base_name, images_dir):
    """Find the actual image file with matching name (could be .jpg, .png, etc.)"""
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        img_path = os.path.join(images_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None

def extract_smoke_images(labels_dir, images_dir, output_dir, split_name):
    """Extract all smoke images from a split (train/val)"""
    count = 0
    
    # Get all label files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    print(f"\nProcessing {split_name} split: {len(label_files)} label files found")
    
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        
        # Check if label file is not empty (has annotations = has smoke)
        if os.path.getsize(label_path) > 0:
            base_name = os.path.splitext(label_file)[0]
            img_path = get_image_extension(base_name, images_dir)
            
            if img_path:
                output_path = os.path.join(output_dir, f"{base_name}_{split_name}{os.path.splitext(img_path)[1]}")
                shutil.copy2(img_path, output_path)
                count += 1
                
                if count % 100 == 0:
                    print(f"  Copied {count} images...")
    
    print(f"  ✓ Total smoke images from {split_name}: {count}")
    return count

# Extract from both train and val splits
train_count = extract_smoke_images(train_labels_dir, train_images_dir, smoke_output_dir, "train")
val_count = extract_smoke_images(val_labels_dir, val_images_dir, smoke_output_dir, "val")

total_count = train_count + val_count
print(f"\n✓ All done! Total smoke images saved: {total_count}")
print(f"Location: {smoke_output_dir}")
