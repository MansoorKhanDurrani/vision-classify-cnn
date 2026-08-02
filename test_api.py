import requests

image_path = "data/seg_test/seg_test/mountain/20058.jpg"

with open(image_path, "rb") as f:
    response = requests.post("http://127.0.0.1:5000/predict", files={"image": f})

print(response.json())