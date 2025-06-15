import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Configurações globais
CLASS_NAMES = ["Hemorragico", "Isquemico", "Normal"]
CLASS_COLORS = {
    0: (255, 0, 0),   # Vermelho: Hemorrágico
    1: (0, 0, 255),   # Azul: Isquêmico
    2: (0, 255, 0)    # Verde: Normal
}

MAX_SIZE = 640
CONFIDENCE_THRESHOLD = 0.6

# Sessão global do ONNX
_session = None

def initialize_onnx(onnx_path):
    global _session
    if _session is None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        _session = ort.InferenceSession(
            onnx_path, 
            options,
            providers=['CPUExecutionProvider']
        )
    return _session

def run_yolo_onnx_inference(model_path, image_path, output_path):
    # Carregar imagem
    image = cv2.imread(image_path)
    original_image = image.copy()
    H, W, _ = image.shape
    
    # Pré-processamento (YOLOv8 específico)
    input_tensor ,(ratio, pad), origin_shape = prepare_input(image)
    
    # Carregar modelo ONNX
    session = initialize_onnx("./model/steve.onnx")
    input_name = session.get_inputs()[0].name
    
    # Inferência
    outputs = session.run(None, {input_name: input_tensor})
    # Processar saídas
    detections = process_output(outputs, H, W, ratio, pad)
    
    # Desenhar bounding boxes
    if len(detections) > 0:
        image_with_boxes = draw_boxes(original_image, detections)
        cv2.imwrite(output_path, image_with_boxes)
        print(f"Detecções salvas em: {output_path}")
        return True
    else:
        print("Nenhuma detecção encontrada")
        cv2.imwrite(output_path, original_image)
        return False

def prepare_input(image, target_size=640):

    h, w = image.shape[:2]
    
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)  # ERRADO!
    
    # SOLUÇÃO: Criar canvas com tamanho fixo target_size x target_size
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # CORRETO
    
    # Calcular posição para centralizar - CORRETO
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    
    # PROBLEMA 2: Aqui você está tentando colocar a imagem redimensionada
    # em um canvas que tem o mesmo tamanho da imagem redimensionada
    canvas[dy:dy+new_h, dx:dx+new_w] = resized  
   
    input_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
    # Reorganizar dimensões: HWC -> CHW
    input_img = input_img.transpose(2, 0, 1)
    
    # Adicionar dimensão de batch e normalizar
    input_img = np.expand_dims(input_img, 0).astype(np.float32) / 255.0
    
    # Verificação crítica de dimensões
    if input_img.shape != (1, 3, target_size, target_size):
        # CORREÇÃO DE EMERGÊNCIA: Forçar redimensionamento se necessário
        fixed_img = cv2.resize(canvas, (target_size, target_size))
        fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_BGR2RGB)
        fixed_img = fixed_img.transpose(2, 0, 1)
        input_img = np.expand_dims(fixed_img, 0).astype(np.float32) / 255.0
        print(f"AVISO: Dimensões corrigidas de {input_img.shape} para (1, 3, {target_size}, {target_size})")
    # Isso não faz sentido se canvas = resized
    return input_img, (scale, (dx, dy)), (h, w)
    # input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # input_img = cv2.resize(input_img, (640, 640))
    # input_img = input_img.transpose(2, 0, 1)  # HWC -> CHW
    # input_img = np.expand_dims(input_img, 0)  # Adicionar dimensão batch
    # input_img = input_img.astype(np.float32) / 255.0
    #return input_img

def process_output(output, img_height, img_width, ratio, pad):
    # YOLOv8 ONNX tem formato de saída: [batch, 84, 8400]
    predictions = np.squeeze(output[0]).T
    
    # Filtrar detecções por confiança
    conf_threshold = 0.5
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]
    
    if len(scores) == 0:
        return []
    
    # Obter classes
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    
    # Obter bounding boxes (formato cxcywh normalizado)
    boxes = predictions[:, :4]
    
    # Desnormalizar coordenadas
    input_shape = np.array([640, 640, 640, 640])
    img_shape = np.array([img_width, img_height, img_width, img_height])
    boxes = np.divide(boxes, input_shape)
    boxes = np.multiply(boxes, img_shape)
    
    # Converter cxcywh para xyxy
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    boxes = [
        [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        for x, y, w, h in zip(x_center, y_center, width, height)
    ]

    left_pad, top_pad = pad
    boxes = [
        [
            max(0, (x1 - left_pad) / ratio),
            max(0, (y1 - top_pad) / ratio ), 
            min(img_width, ( x2 - left_pad ) / ratio ),
            min(img_height, ( y2 - top_pad ) / ratio),
        ]
        for (x1, y1, x2, y2) in boxes
    ]
    
    # Aplicar NMS
    nms_threshold = 0.5
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes, 
        scores=scores.tolist(), 
        score_threshold=conf_threshold, 
        nms_threshold=nms_threshold
    )
    
    if len(indices) > 0:
        detections = []
        for i in indices.flatten():
            detections.append({
                "class_id": class_ids[i],
                "confidence": scores[i],
                "box": boxes[i]
            })
        return detections
    
    return []

def draw_boxes(image, detections):
    
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        class_id = det["class_id"]
        confidence = det["confidence"]
        
        # Converter para inteiros
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Desenhar bounding box
        color = CLASS_COLORS.get(class_id, (255, 255, 0))
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Criar label
        label = f"{CLASS_NAMES[class_id]}: {confidence:.2f}"
        
        # Calcular posição do texto
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        
        # Desenhar fundo para texto
        cv2.rectangle(
            image, 
            (x1, text_y - text_height - 4),
            (x1 + text_width, text_y + 4), 
            color, 
            -1
        )
        
        # Adicionar texto
        cv2.putText(
            image, 
            label, 
            (x1, text_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )
    
    return image

def process_image(image, session):

    img_cv = np.array(image)
    if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:  # RGBA para RGB
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
    else:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    input_tensor, (scale, pad), origin_shape = prepare_input(img_cv)
    input_name = session.get_inputs()[0].name    
    
    outputs = session.run(None, {input_name: input_tensor})

    detections = process_output(outputs, origin_shape[0], origin_shape[1], scale, pad)

    if detections:
        result_image = draw_boxes(img_cv, detections)
    else:
        result_image = image
    
    if isinstance( result_image, np.ndarray):
        result_image = Image.fromarray(result_image)
    return result_image, detections
