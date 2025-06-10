import os

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from ultralytics import YOLO
from flasgger import Swagger 
from PIL import Image


RESULTS_DIR = 'results'
MODEL_DIR = 'model'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR , exist_ok=True)
app = Flask(__name__)
CORS(app)
# Configuração do Swagger
app.config['SWAGGER'] = {
    'title': 'API de Detecção de Objetos com YOLO',
    'description': 'API para detecção de objetos em imagens usando YOLO',
    'version': '1.0',
    'uiversion': 3,
    'specs_route': '/docs/'  
}
swagger = Swagger(app)


def loadModel():
  return  YOLO('model/steve.pt');

@app.route("/")
def index():
  return "Index API esta funcionando"

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
  """
    Endpoint para detecção de objetos em imagens usando YOLO
    ---
    tags:
      - Detecção
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Imagem para detecção de objetos
    responses:
      200:
        description: Resultado da detecção
        schema:
          type: object
          properties:
            filename:
              type: string
            img_bytes:
              type: string
            size_bytes:
              type: integer
      400:
        description: Erro quando a imagem não é fornecida
    """

  if "file"  not in request.files:
    return jsonify({'error':'Imagem não encontrada'}), 400

  files = request.files.getlist('file')

  model = loadModel()
  result_list = []
  for img_file in files:
    img = Image.open(img_file.stream).convert("RGB")
    results = model(img)
    output_path = os.path.join(RESULTS_DIR, f"pred_{img_file.filename}")
    results[0].save(filename=output_path)

    with open(output_path, "rb") as f:
      img_bytes = f.read()

    result_list.append({
      'filename': img_file.filename,
      'img_bytes': f"/output/{img_file.filename}",
      'size_bytes': len(img_bytes)
    })

  return jsonify(result_list )
  
@app.route('/output/<filename>')
@cross_origin()
def get_output(filename):
  """
    Retorna a imagem com as detecções feitas pelo modelo YOLO
    ---
    tags:
      - Resultado
    parameters:
      - name: filename
        in: path
        type: string
        required: true
        description: Nome do arquivo de imagem gerado pela predição
    responses:
      200:
        description: Imagem com bounding boxes desenhados
        content:
          image/jpeg:
            schema:
              type: string
              format: binary
      400:
        description: Imagem não encontrada
    """
  path = os.path.join(RESULTS_DIR, f"pred_{filename}")
  if not os.path.exists(path):
    return jsonify({'error':'Imagem não encontrada'}), 400
  return send_file(path, mimetype="image/jpeg")
if __name__ == "__main__":
  app.run(
    debug=True, 
    host="0.0.0.0", 
    port=int(os.environ.get("PORT", 3000))
  )