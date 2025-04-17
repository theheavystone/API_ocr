import os
import openai
from PIL import Image
import base64
import json
from dotenv import load_dotenv


# OpenAI API setup
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# image directory setup
image_dir = "images/"
output_dir = "ocr_results/"
os.makedirs(output_dir, exist_ok=True)

# Get a list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Load image as base64 encoded (only if necessary)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# image batch processing
for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)

    # Send image to OpenAI API for OCR
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "この画像のテキストをすべて抽出するのを手伝ってください。"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(img_path)}"
                    }
                }
            ]}
        ],
        max_tokens=1024
    )

    # result output
    # print(f"\n📄 Image: {img_file}")
    # print(response.choices[0].message.content.strip())

     # extract OCR text from the response
    ocr_text = response.choices[0].message.content.strip()

    # save the OCR text to a JSON file
    output_data = {
        "filename": img_file,
        "ocr_text": ocr_text
        }
    output_path = os.path.join(output_dir, img_file + ".json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Save complete: {output_path}")
