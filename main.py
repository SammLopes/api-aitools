import gc
import time
import os
import torch
import logging
from swagger_config import SWAGGER_CONFIG
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from ultralytics import YOLO
from flasgger import Swagger, swag_from 
from PIL import Image

# Configurações de otimização extrema para máquinas pequenas
torch.set_num_threads(1)  # Uma thread apenas
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'  # Intel MKL
torch.set_grad_enabled(False)  # Desabilita gradientes globalmente

RESULTS_DIR = 'results'
MODEL_DIR = 'model'

CONFIDENCE = 0.6  
MAX_SIZE = 640       
MAX_DETECTIONS = 50  

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR , exist_ok=True)

app = Flask(__name__)
CORS(app)

app.config['SWAGGER'] = SWAGGER_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs.txt")
    ]
)

swagger = Swagger(app)

_model = None
_model_load_time = None

def loadModel():
  global _model, _model_load_time

  if _model is None:
    start_time = time.time()

    onnx_path = os.path.join(MODEL_DIR, 'steve.onnx')
    if os.path.exists(onnx_path):
      _model = YOLO(onnx_path)
      model_type = "ONNX"
    else:
      raise FileNotFoundError("Modelo não encontrado! Coloque steve.onnx ou steve.pt na pasta model/")
    
    _model_load_time = time.time() - start_time
    logging.info(f"✅ Modelo {model_type} carregado em {_model_load_time:.2f}s");
  
    gc.collect()

  return _model

@app.route("/")
def index():
  """Health check básico"""
  return jsonify({
      "status": "API YOLO ONNX funcionando",
      "model_loaded": _model is not None,
      "endpoints": [
        "/predict", 
        "/output/<filename>", 
        "/docs/",
        "/cleanup",
        "/status"
      ]
  })

@app.route("/predict", methods=["POST"])
@cross_origin()
@swag_from("docs/predict.yaml")
def predict(): 
  start_time = time.time()
  
  if "file"  not in request.files:
    return jsonify({'error':'Imagem não encontrada'}), 400

  files = request.files.getlist('file')

  if len(files) > 3:
        return jsonify({'error': 'Máximo 3 imagens por requisição (limite para máquinas pequenas)'}), 400
    
  model = loadModel()
  processing_times = []
  result_list = []

  for idx, img_file in enumerate(files):
      
    img_start = time.time()
    
    img_file.seek(0, 2)  # Vai para o final
    file_size = img_file.tell()
    img_file.seek(0)     # Volta para o início

    if file_size > 5 * 1024 * 1024:
      return jsonify({
          'error': f'Arquivo {img_file.filename} muito grande ({file_size/1024/1024:.1f}MB). Máximo: 5MB'
      }), 400

    img = Image.open(img_file.stream).convert("RGB")
    original_size = img.size
    if max(img.size) > MAX_SIZE:

      ratio = MAX_SIZE / max(img.size)
      new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
      img = img.resize(new_size, Image.Resampling.LANCZOS)
      print(f"🔄 Imagem {idx+1} redimensionada: {original_size} → {img.size}")

    pred_start = time.time()

    results = model(
      img,
      conf=CONFIDENCE,
      verbose=False,
      save=False,
      show=False,
      max_det = MAX_DETECTIONS,
      device="cpu",
      imgsz=MAX_SIZE
    )
    pred_time = time.time() - pred_start

    output_path = os.path.join(RESULTS_DIR, f"pred_{img_file.filename}")
    print(results)
    results[0].save(filename=output_path)

    with open(output_path, "rb") as f:
      img_bytes = f.read()

    detections_count = len(results[0].boxes) if results[0].boxes is not None else 0
    img_total_time = time.time() - img_start
    processing_times.append({
        'prediction_time': pred_time,
        'total_time': img_total_time
    })

    result_list.append({
      'filename': img_file.filename,
      'img_bytes': f"/output/{img_file.filename}",
      'size_bytes': len(img_bytes),
      'detections_count': detections_count 
    })

    del img, results
  
  gc.collect()    

  total_time = time.time() - start_time
  avg_pred_time = sum(p['prediction_time'] for p in processing_times) / len(processing_times)
  
  for i, p in enumerate(processing_times):
    logging.info(f"[{files[i].filename}] ⏱️ Tempo de predição: {p['prediction_time']:.3f}s | Tempo total img: {p['total_time']:.3f}s")

  logging.info(f"📦 Requisição com {len(files)} imagem(s) processada(s)")
  logging.info(f"🧮 Média de tempo por imagem: {avg_pred_time:.3f}s | Tempo total da requisição: {total_time:.3f}s\n")

  for i, p in enumerate(processing_times):
    filename = files[i].filename
    detections = result_list[i].get('detections_count', '?')  # adicionaremos isso já já
    logging.info(
        f"[{filename}] 📸 {detections} detecções | "
        f"⏱️ Predição: {p['prediction_time']:.3f}s | "
        f"Total imagem: {p['total_time']:.3f}s"
    )

  return jsonify( result_list )
  
@app.route('/output/<filename>')
@cross_origin()
@swag_from("docs/output.yaml")
def get_output(filename):
  path = os.path.join(RESULTS_DIR, f"pred_{filename}")
  if not os.path.exists(path):
    return jsonify({'error':'Imagem não encontrada'}), 400
  return send_file(path, mimetype="image/jpeg")

@app.route('/cleanup', methods=['POST'])
@cross_origin()  
@swag_from("docs/cleanup.yaml")
def cleanup():
  try:
    removed_files = 0
    for filename in os.listdir(RESULTS_DIR):
        file_path = os.path.join(RESULTS_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            removed_files += 1
      
    gc.collect()
    
    return jsonify({
        'success': True,
        'message': 'Limpeza concluída'
    })
  except Exception as e:
      return jsonify({'error': f'Erro na limpeza: {str(e)}'}), 500

@app.route('/status', methods=['GET'])  
@cross_origin()
@swag_from("docs/status.yaml")
def status():
 
  arquivos = os.listdir(RESULTS_DIR )
  total_arquivos  = [ f for f in arquivos if os.path.isfile( os.path.join(RESULTS_DIR, f) ) ]

  return jsonify({
    "total_arquivos": len(total_arquivos)
  })

if __name__ == "__main__":
  app.run(
    debug=True, 
    host="0.0.0.0", 
    port=int(os.environ.get("PORT", 3000))
  )