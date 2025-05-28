import os

from flask import Flask
from ultralytics import YOLO

app = Flask(__name__)

def loadModel():
  return  YOLO('./model');

@app.route("/")
def index():
  return "Index API esta funcionando"

@app.route("/predict")
def predict():
  if "image" not in request.files:
    return jsonify({'error':'Imagem não encontrada'}), 400

  img_file = request.files['image']
  img = Image.open(img_file.stream)
  
  model = loadModel()
  results = model(img)

  boxes = results[0].boxes.data.cpu().numpy()
  class_names = model.name

  predictions = []
  for box in boxes:
    x1, y1, x2, y2, conf, cls = box
    predictions.append({
      'class': class_names[int(cls)],
      'confidence': float(conf),
      'bbox': [int(x1), int(y1), int(x2), int(y2)]
    })
  
  return jsonify(predictions)
  
if __name__ == "__main__":
  app.run(
    debug=True, 
    host="0.0.0.0", 
    port=int(os.environ.get("PORT", 3000))
  )