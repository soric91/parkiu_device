
from src.parking_360.detection.detector import ParkingSpaceDetector
from src.parking_360.camera.img_convert import SectorizadorImagen 

import json


angulos = {
    "Norte": (65, 165),
    "Sur": (235, 315),
    "Este": (315, 65),
    "Oeste": (155, 230)
}

path_img = "src/parking_360/res/img_1.jpg"
model = "yolov8n.pt"
config_celda = json.load(open("src/parking_360/detection/parking_config.json"))


def main():
    sectorizador = SectorizadorImagen(path_img, angulos)
    dectetor = ParkingSpaceDetector(model)
    parking_space = dectetor.convert_rotated_rects_to_parking_spaces(config_celda)
    sectores_para_yolo = sectorizador.obtener_sectores()

    resultados = dectetor.analyze_parking_spaces(sectores_para_yolo, parking_space)
    
    dectetor.visualize_results(sectores_para_yolo,parking_space , resultados)
    
    print("Resultados de la detección de espacios de estacionamiento:")
    print(resultados.to_dict())
    
if __name__ == "__main__":
    main()
