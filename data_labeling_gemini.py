import json
import os
from pathlib import Path

from google import genai


INPUT_PATH = Path("data.jsonl")
OUTPUT_PATH = Path("data_gemini_labeled.jsonl")
CHECKPOINT_PATH = Path("data_gemini_labeled.checkpoint")

PROMPT = """You are tasked with detecting wildfire smoke in images.

Question: Is there smoke in this image coming from burning vegetation (trees, forest, grass)?

Respond with ONLY valid JSON in this exact format, no markdown, no extra text:
{
"reasoning": "Your analysis of why this is true or false (mention presence/absence of smoke, characteristics, confidence). Keep the reasoning brief, ideally one sentence.",
"decision": true or false
}

Return ONLY the JSON object. true = smoke from vegetation, false = no smoke or non-vegetation smoke (chimney, clouds, industrial)."""


def build_gemini_prompt(label: str) -> str:
    if label == "A":
        return (
            "The image contains smoke from burning vegetation. "
            "Answer the following prompt like normal with this info in mind: "
            + PROMPT
        )
    if label == "B":
        return (
            "The image does contain smoke but not from burning vegetation. "
            "Answer the following prompt like normal with this info in mind: "
            + PROMPT
        )
    if label == "C":
        return (
            "The image does not contain smoke. "
            "Answer the following prompt like normal with this info in mind: "
            + PROMPT
        )
    raise ValueError(f"Unexpected label: {label}")


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_checkpoint() -> int:
    if CHECKPOINT_PATH.exists():
        raw = CHECKPOINT_PATH.read_text(encoding="utf-8").strip()
        return int(raw) if raw else 0
    # Fallback: if checkpoint is missing but output exists, resume by output size.
    return count_jsonl_lines(OUTPUT_PATH)


def write_checkpoint(next_index: int) -> None:
    CHECKPOINT_PATH.write_text(str(next_index), encoding="utf-8")


def main() -> None:
    # api_key = os.getenv("GEMINI_API_KEY")
    # if not api_key:
    #     raise RuntimeError("Set GEMINI_API_KEY in your environment before running this script.")

    client = genai.Client(api_key="")

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    start_idx = read_checkpoint()
    print(f"Resuming at index {start_idx} of {len(data)}")

    with OUTPUT_PATH.open("a", encoding="utf-8") as out_f:
        for idx, line in enumerate(data):
            if idx < start_idx:
                continue

            line["messages"][0]["content"][1]["text"] = PROMPT
            label = line["messages"][1]["content"][0]["text"]
            image = line["messages"][0]["content"][0]["image"]
            gemini_prompt = build_gemini_prompt(label)

            try:
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=[image, gemini_prompt],
                )
                line["messages"][1]["content"][0]["text"] = response.text
            except Exception as exc:
                print(f"Stopped at index {idx}: {exc}")
                print("Rerun the script to continue from the checkpoint.")
                break

            out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            write_checkpoint(idx + 1)
            print(f"Processed index {idx}")
        else:
            print("Finished all rows.")


if __name__ == "__main__":
    main()


# input_folder = r"U:\Fraunhofer Waldbrand\Testbilder\TrainingDataset\forestfire_0326"
# for image_path in Path(input_folder).iterdir():
#     if image_path.is_file() and image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
#         my_file = client.files.upload(file=r"U:\Fraunhofer Waldbrand\Testbilder\TrainingDataset\CroppedData\No_smoke\20260323_091626_(110.04,0.0,754.0)_yes_mask.jpg")

#         response = client.models.generate_content(
#             model="gemini-3-flash-preview",
#             contents=[my_file, """The image does not contain smoke. Answer the following prompt like normal with this info in mind. You are tasked with detecting wildfire smoke in images.

#         Question: Is there smoke in this image coming from burning vegetation (trees, forest, grass)?

#         Respond with ONLY valid JSON in this exact format, no markdown, no extra text:
#         {
#         "reasoning": "Your analysis of why this is true or false (mention presence/absence of smoke, characteristics, confidence). Keep the reasoning brief, ideally one sentence.",
#         "decision": true or false
#         }

#         Return ONLY the JSON object. true = smoke from vegetation, false = no smoke or non-vegetation smoke (chimney, clouds, industrial)."""],
#         )


#         print(response.text)
#         break
