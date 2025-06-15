import gc
import time
import os
import logging
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from flasgger import Swagger, swag_from 
from swagger_config import SWAGGER_CONFIG
from onnx import initialize_onnx, process_image

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 

RESULTS_DIR = 'results'
MODEL_DIR = 'model'

CONFIDENCE = 0.6  
MAX_SIZE = 640       
MAX_DETECTIONS = 50  
MAX_FILES_PER_REQUEST = 5
MAX_FILE_SIZE = 5 * 1024 * 1024  

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

onnx_path = os.path.join(MODEL_DIR, 'steve.onnx')
if not os.path.exists(onnx_path):
    logging.error("Modelo não encontrado! Coloque steve.onnx na pasta model/")
    exit(1)
  
session = initialize_onnx(onnx_path)

@app.route("/")
def index():
  """Health check básico"""
  return jsonify({
      "status": "API YOLO ONNX funcionando",
      "model_loaded": session is not None,
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

  if len(files) > MAX_FILES_PER_REQUEST:
        return jsonify({'error': f'Máximo{MAX_FILES_PER_REQUEST} imagens por requisição (limite para máquinas pequenas)'}), 400
    
  processing_times = []
  result_list = []

  for idx, img_file in enumerate(files):
      
    img_start = time.time()
    
    img_file.seek(0, 2)  
    file_size = img_file.tell()
    img_file.seek(0)     

    if file_size > MAX_FILE_SIZE:
      return jsonify({
          'error': f'Arquivo {img_file.filename} muito grande ({file_size/1024/1024:.1f}MB). Máximo: 5MB'
      }), 400

    img = Image.open(img_file.stream).convert("RGB")
    img_process_start = time.time()

    result_img, detections = process_image(img, session)

    process_time = time.time() - img_process_start

    output_path = os.path.join(RESULTS_DIR, f"pred_{img_file.filename}")
    result_img.save(output_path)

    img_bytes = BytesIO()
    result_img.save(img_bytes, format='JPEG')
    img_size = img_bytes.tell()

    img_total_time = time.time() - img_start
    processing_times.append({
        'processing_time': img_total_time,
        'total_time': img_total_time
    })

    result_list.append({
      'filename': img_file.filename,
      'img_bytes': f"/output/{img_file.filename}",
      'size_bytes': img_size,
      'detections_count': len(detections)
    })

    del img, result_img
    gc.collect()    

  total_time = time.time() - start_time
  logging.info(f"📦 Requisição com {len(files)} imagem(s) processada(s) em {total_time:.2f}s")

  for i, times in enumerate(processing_times):
      filename = files[i].filename
      detections = result_list[i]['detections_count']

      proc_time = times['processing_time']  
      total_time_img = times['total_time']
      
      logging.info(
          f"[{filename}] 🔍 {detections} detecções | "
          f"⏱️ Processamento: {proc_time:.3f}s | "
          f"Total: {total_time_img:.3f}s"
      )    
  return jsonify(result_list)
  
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
  print("PORT =", os.environ.get("PORT"))
  app.run(
    debug=True, 
    host="0.0.0.0", 
    port=int(os.environ.get("PORT", 3000))
  )