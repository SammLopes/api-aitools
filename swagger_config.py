SWAGGER_CONFIG = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,  # todas as rotas
            "model_filter": lambda tag: True,  # todos os tags
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/",
    "title": "YOLO ONNX API",
    "version": "2.0",
    "description": "API otimizada para detecção de objetos usando YOLO com ONNX",
}
