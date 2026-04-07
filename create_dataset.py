import json
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import io
import csv
from datasets import load_dataset
import torch
from PIL import Image

no_smoke = r"U:\Fraunhofer Waldbrand\Testbilder\TrainingDataset\CroppedData\Train\No_smoke"
chimney_cloud_fog_industrial = r"U:\Fraunhofer Waldbrand\Testbilder\TrainingDataset\CroppedData\Train\Chimney_fog"
fire = r"U:\Fraunhofer Waldbrand\Testbilder\TrainingDataset\CroppedData\Train\Fire"
with open("data.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "question", "answer"])
    for image_path in Path(no_smoke).iterdir():
        if image_path.is_file() and image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            writer.writerow([str(image_path),"""You are tasked with detecting wildfire smoke.

                Determine:
                Is there smoke in the image coming from burning vegetation (trees, forest, grass)?

                Answer with ONLY the letter:
                A: yes, Smoke is present and is coming from burning vegetation
                B: no (e.g., there is no smoke, or there is smoke but it's from a non-vegetation source like a chimney or industrial)
                c: it can't be determined whether there is smoke or its source""","C"])
    for image_path in Path(chimney_cloud_fog_industrial).iterdir():
        if image_path.is_file() and image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            writer.writerow([str(image_path),"""You are tasked with detecting wildfire smoke.

                Determine:
                Is there smoke in the image coming from burning vegetation (trees, forest, grass)?

                Answer with ONLY the letter:
                A: yes, Smoke is present and is coming from burning vegetation
                B: no (e.g., there is no smoke, or there is smoke but it's from a non-vegetation source like a chimney or industrial)
                C: it can't be determined whether there is smoke or its source""","B"])
    for image_path in Path(fire).iterdir():
        if image_path.is_file() and image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            writer.writerow([str(image_path),"""You are tasked with detecting wildfire smoke.

                Determine:
                Is there smoke in the image coming from burning vegetation (trees, forest, grass)?

                Answer with ONLY the letter:
                A: yes, Smoke is present and is coming from burning vegetation
                B: no (e.g., there is no smoke, or there is smoke but it's from a non-vegetation source like a chimney or industrial)
                C: it can't be determined whether there is smoke or its source""","A"])
            
df = pd.read_csv("data.csv")

def format_data(sample):
    return {
        "messages": [
            # {
            #     "role": "system",
            #     "content": [
            #         {
            #             "type": "text",
            #             "text": "You are a strict image classifier. Always answer with exactly one letter: A or B. Do not explain your answer."
            #         }
            #     ],
            # },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["image_path"],
                    },
                    {
                        "type": "text",
                        "text": sample["question"],
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": str(sample["answer"])
                    }
                ],
            },
        ]
    }

with open("data.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        formatted_sample = format_data(row.to_dict())
        f.write(json.dumps(formatted_sample) + "\n")

# output_folder_smoke = Path(r"U:\Fraunhofer Waldbrand\Testbilder\ForestFireInsights-EvalImages\smoke")
# output_folder_nosmoke = Path(r"U:\Fraunhofer Waldbrand\Testbilder\ForestFireInsights-EvalImages\no_smoke")
# # Create output folder if it doesn't exist
# os.makedirs(output_folder_smoke, exist_ok=True)
# os.makedirs(output_folder_nosmoke, exist_ok=True)

# df = pd.read_parquet("hf://datasets/leon-se/ForestFireInsights-Eval/data/train-00000-of-00001.parquet")

# i=1
# for row in df.iloc:
#     img_bytes = row.image["bytes"]
#     image = Image.open(io.BytesIO(img_bytes))
#     if '"forest_fire_smoke_visible": "Yes"' in row["gt_answer"]:
#         image.save(output_folder_smoke / f"smoke_{i}.jpg")
#     else:
#         image.save(output_folder_nosmoke / f"no_smoke_{i}.jpg")
#     i+=1