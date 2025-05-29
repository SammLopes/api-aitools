import os

from flask import Flask, request, jsonify
from ultralytics import YOLO
from flasgger import Swagger 

app = Flask(__name__)
# Configuração do Swagger
app.config['SWAGGER'] = {
    'title': 'API de Detecção de Objetos com YOLO',
    'description': 'API para detecção de objetos em imagens usando YOLO',
    'version': '1.0',
    'uiversion': 3,
    'specs_route': '/docs/'  # Rota para acessar a documentação Swagger UI
}
swagger = Swagger(app)


def loadModel():
  return  YOLO('./model');

@app.route("/")
def index():
  return "Index API esta funcionando"

@app.route("/predict", methods=["POST"])
def predict():
  """
    Endpoint para detecção de objetos em imagens usando YOLO
    ---
    tags:
      - Detecção
    consumes:
      - multipart/form-data
    parameters:
      - name: image
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
      400:
        description: Erro quando a imagem não é fornecida
    """
  if "image" not in request.files:
    return jsonify({'error':'Imagem não encontrada'}), 400

  img_file = request.files.getlist('images')
  img = Image.open(img_file.stream)
  
  model = loadModel()
  results = model(img)

  boxes = results[0].boxes.data.cpu().numpy()
  class_names = model.name

  result_list = []
  for img_file in files:
    img = Image.open(img_file.stream).convert("RGB")
    results = model(img)
    results[0].save(filename=f"pred_{img_file.filename}")

    with open(f"pred_{img_file.filename}", "rb") as f:
      img_bytes = f.read()
    img_io = io.BytesIO(img_bytes)
    result_list.append({
      'filename':img_file.filename,
      'img_bytes': f"/output/{img_file.filename}"
    })
  
  
  return jsonify(result_list )
  
@app.route('/output/<filename>')
def get_output(filename):
  path = f"pred_{filename}"
  if not os.path.exists(path):
    return jsonify({'error':'Imagem não encontrada'}), 400
  return send_file(path, mimetype="image/jpeg")
if __name__ == "__main__":
  app.run(
    debug=True, 
    host="0.0.0.0", 
    port=int(os.environ.get("PORT", 3000))
  )