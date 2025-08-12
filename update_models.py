import requests

text_url = "https://text.pollinations.ai/models"
image_url = "https://image.pollinations.ai/models"
response_text = requests.get(text_url)
response_image = requests.get(image_url)
response_text.raise_for_status()
response_image.raise_for_status()


def update_model():
    try :
        with open("text_models.json", "w", encoding="utf-8") as f:
            f.write(response_text.text)
            f.close()
    except Exception as e:
        print(f"Error writing text models: {e}")
        return
    try:
        with open("image_models.json", "w", encoding="utf-8") as f:
            f.write(response_image.text)
            f.close()
    except Exception as e:
        print(f"Error writing image models: {e}")
        return

if __name__ == "__main__":
    update_model()
    print("Text models updated successfully.")