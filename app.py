from flask import Flask, render_template, request, jsonify
import os
from ultralytics import YOLO
import uuid
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load your trained model
model = YOLO("best.pt")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("image")

        if not file:
            return jsonify({"error": "No image uploaded"})

        # Save file
        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # ✅ Resize image (better detection)
        img = cv2.imread(filepath)
        img = cv2.resize(img, (640, 640))
        cv2.imwrite(filepath, img)

        # ✅ Better detection settings
        results = model(filepath, conf=0.05, iou=0.4)

        names = model.names
        boxes = results[0].boxes

        # ✅ Count coins
        coin_counts = {}
        for name in names.values():
            coin_counts[name] = 0

        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            coin_counts[label] += 1

        total_coins = len(boxes)

        # ✅ Coin value mapping
        value_map = {
            "One Rupee": 1,
            "Two Rupee": 2,
            "Two Rupees": 2,
            "Five Rupee": 5,
            "Five Rupees": 5,
            "Ten Rupee": 10,
            "Ten Rupees": 10,
            "Twenty Rupee": 20,
            "Twenty Rupees": 20
        }

        total_amount = 0
        for label, count in coin_counts.items():
            if label in value_map:
                total_amount += value_map[label] * count

        # Save output image
        output_name = "out_" + filename
        output_path = os.path.join(UPLOAD_FOLDER, output_name)

        annotated = results[0].plot()
        cv2.imwrite(output_path, annotated)

        return jsonify({
            "count": total_coins,
            "coins": coin_counts,
            "total": total_amount,
            "image": "/" + output_path
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)