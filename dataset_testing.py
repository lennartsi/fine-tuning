from datasets import load_dataset
import os

# Debug: Check directory structure
data_dir = r"U:\Fraunhofer Waldbrand\datasets\SAINetset_v8.0\data"
print(f"Attempting to load from: {data_dir}")
print(f"Directory exists: {os.path.exists(data_dir)}")
print(f"Contents: {os.listdir(data_dir)}")

print("Starting dataset load...")
ds = load_dataset(
    "imagefolder",
    data_dir=data_dir
)
print("Dataset loaded successfully!")

# example of pictures with smoke from burning vegetation
print(ds["train"][0])
print(ds["train"][0]["messages"][0]["content"][1]["text"])