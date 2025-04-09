from dataclasses import dataclass, field
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

@dataclass
class ParkingSpace:
    """Representa un espacio de estacionamiento"""
    id: str
    box: List[int]  # [x1, y1, x2, y2]
    rotated_box: np.ndarray  # Los 4 puntos del rectángulo rotado
    rect_cv: Tuple  # Formato OpenCV: ((cx, cy), (w, h), angle)
    original: Dict  # Datos originales del rectángulo

@dataclass
class DetectionResult:
    """Resultados de detección para un sector"""
    espacios: Dict[str, bool] = field(default_factory=dict)
    
    def ocupados(self) -> List[str]:
        """Retorna la lista de espacios ocupados"""
        return [id for id, ocupado in self.espacios.items() if ocupado]
    
    def libres(self) -> List[str]:
        """Retorna la lista de espacios libres"""
        return [id for id, ocupado in self.espacios.items() if not ocupado]

@dataclass
class ParkingDetectionResults:
    """Resultados de detección para todos los sectores"""
    sectores: Dict[str, DetectionResult] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Dict[str, bool]]:
        """Convierte los resultados al formato deseado"""
        return {
            sector: results.espacios 
            for sector, results in self.sectores.items()
        }

@dataclass
class ParkingSpaceDetector:
    """
    Detector de espacios de estacionamiento utilizando rectángulos rotados
    """
    model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.25
    model: Optional[Any] = None
    vehicle_classes: List[int] = field(default_factory=lambda: [2, 5, 7])  # car, bus, truck
    
    def __post_init__(self):
        """Inicializa el modelo YOLO si está disponible"""
        try:
            from ultralytics import YOLO
            print(f"Cargando modelo YOLO desde: {self.model_path}")
            self.model = YOLO(self.model_path)
        except ImportError:
            print("ADVERTENCIA: No se pudo importar YOLO. Se utilizará solo detección visual.")
            self.model = None
    
    def convert_rotated_rects_to_parking_spaces(self, rect_data: dict) -> Dict[str, List[ParkingSpace]]:
        """
        Convierte rectángulos rotados a espacios de estacionamiento
        
        Args:
            rect_data: Diccionario con rectángulos rotados por sector
            
        Returns:
            Diccionario con espacios de estacionamiento por sector
        """
        parking_spaces = {}
        
        for direction, spaces in rect_data.items():
            parking_spaces[direction] = []
            
            for space_id, rect in spaces.items():
                # Extraer datos del rectángulo
                cx, cy = rect["cx"], rect["cy"]
                w, h = rect["w"], rect["h"]
                angle = rect["angle"]
                
                # Convertir a formato de rectángulo rotado para OpenCV
                rect_cv = ((cx, cy), (w, h), angle)
                
                # Obtener los 4 puntos del rectángulo
                box = cv2.boxPoints(rect_cv)
                box = np.int32(box)
                
                # Encontrar los límites para crear el bounding box
                x_coords = box[:, 0]
                y_coords = box[:, 1]
                
                x1, y1 = np.min(x_coords), np.min(y_coords)
                x2, y2 = np.max(x_coords), np.max(y_coords)
                
                # Crear espacio de estacionamiento con dataclass
                parking_space = ParkingSpace(
                    id=space_id,
                    box=[x1, y1, x2, y2],
                    rotated_box=box,
                    rect_cv=rect_cv,
                    original=rect
                )
                
                parking_spaces[direction].append(parking_space)
        
        return parking_spaces
    
    def detect_vehicles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detecta vehículos en una imagen usando YOLO
        
        Args:
            image: Imagen en formato numpy array (BGR)
            
        Returns:
            Lista de detecciones con formato:
            [{'box': [x1, y1, x2, y2], 'confidence': conf, 'class': class_id}]
        """
        if self.model is None:
            return []
            
        # Realizar la detección
        results = self.model(image, conf=self.conf_threshold)[0]
        
        detections = []
        
        # Procesar resultados
        for det in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            
            # Verificar si es un vehículo (si usamos modelo estándar)
            if int(cls) in self.vehicle_classes:
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': int(cls)
                })
        
        return detections
    
    def analyze_parking_spaces(self, sectores: Dict[str, np.ndarray], 
                           parking_spaces: Dict[str, List[ParkingSpace]]) -> ParkingDetectionResults:
        """
        Analiza si hay vehículos en los espacios de estacionamiento definidos
        
        Args:
            sectores: Diccionario con las imágenes de los sectores
            parking_spaces: Espacios de estacionamiento por sector
            
        Returns:
            Objeto ParkingDetectionResults con los resultados por sector
        """
        results = ParkingDetectionResults()
        
        for direction, image in sectores.items():
            # Inicializar resultados para esta dirección
            sector_result = DetectionResult()
            
            # Detectar vehículos en esta vista
            detections = self.detect_vehicles(image)
            
            # Obtener espacios para esta dirección
            spaces = parking_spaces.get(direction, [])
            
            for space in spaces:
                # Comprobar si hay algún vehículo en este espacio
                is_occupied = False
                
                # Para cada detección de vehículo
                for detection in detections:
                    vehicle_box = detection['box']
                    
                    # Crear una máscara para el espacio de estacionamiento
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    cv2.drawContours(mask, [space.rotated_box], 0, 255, -1)
                    
                    # Crear una máscara para el vehículo detectado
                    x1, y1, x2, y2 = vehicle_box
                    vehicle_mask = np.zeros_like(mask)
                    cv2.rectangle(vehicle_mask, (x1, y1), (x2, y2), 255, -1)
                    
                    # Calcular la intersección
                    intersection = cv2.bitwise_and(mask, vehicle_mask)
                    intersection_area = cv2.countNonZero(intersection)
                    
                    # Área del espacio de estacionamiento
                    space_area = cv2.countNonZero(mask)
                    
                    # Si hay suficiente solapamiento, considerar ocupado
                    if space_area > 0 and intersection_area / space_area > 0.25:
                        is_occupied = True
                        break
                
                # Guardar en el formato solicitado: true si está ocupado, false si está libre
                sector_result.espacios[space.id] = is_occupied
            
            # Añadir resultados de este sector
            results.sectores[direction] = sector_result
        
        return results
    
    def visualize_results(self, sectores: Dict[str, np.ndarray], 
                        parking_spaces: Dict[str, List[ParkingSpace]],
                        results: ParkingDetectionResults) -> None:
        """
        Visualiza los resultados de la detección
        
        Args:
            sectores: Diccionario con las imágenes de los sectores
            parking_spaces: Espacios de estacionamiento por sector
            results: Resultados del análisis
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        direcciones = ['Norte', 'Sur', 'Este', 'Oeste']
        
        for idx, direccion in enumerate(direcciones):
            img = sectores[direccion].copy()
            spaces = parking_spaces.get(direccion, [])
            
            # Obtener resultados para este sector
            sector_result = results.sectores.get(direccion, DetectionResult())
            
            # Contadores para el título
            ocupados = len(sector_result.ocupados())
            libres = len(sector_result.libres())
            
            for space in spaces:
                # Color según disponibilidad (true = ocupado, false = libre)
                is_occupied = sector_result.espacios.get(space.id, False)
                
                if is_occupied:
                    color = (0, 0, 255)  # Rojo - ocupado
                    status = "Ocupado"
                else:
                    color = (0, 255, 0)  # Verde - libre
                    status = "Libre"
                
                # Dibujar el contorno
                cv2.drawContours(img, [space.rotated_box], 0, color, 2)
                
                # Calcular el centro para el texto
                cx = int(np.mean(space.rotated_box[:, 0]))
                cy = int(np.mean(space.rotated_box[:, 1]))
                
                # Mostrar el ID
                cv2.putText(img, f"{space.id}", (cx-10, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(f"{direccion} - {libres} libres / {ocupados} ocupados")
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        return fig