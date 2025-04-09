import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class SectorizadorImagen:
    ruta_imagen: str
    angulos: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Norte": (45, 135),
        "Sur": (225, 315),
        "Este": (315, 45),
        "Oeste": (135, 225)
    })

    def __post_init__(self):
        self.img = cv2.imread(self.ruta_imagen)
        if self.img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen en: {self.ruta_imagen}")
        self.center = (self.img.shape[1] // 2, self.img.shape[0] // 2)
        self.radius = min(self.center)
        self.sectores = self._generar_sectores()

    def _generar_sectores(self) -> Dict[str, np.ndarray]:
        sectores = {}
        for nombre, (inicio, fin) in self.angulos.items():
            sectores[nombre] = self._crop_sector(inicio, fin)
        return sectores

    def _crop_sector(self, angle_start: float, angle_end: float) -> np.ndarray:
        mask = np.zeros_like(self.img[:, :, 0])
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                dx, dy = x - self.center[0], y - self.center[1]
                r = math.hypot(dx, dy)
                theta = (math.degrees(math.atan2(dy, dx)) + 360) % 360
                if r < self.radius:
                    if angle_start < angle_end:
                        if angle_start <= theta < angle_end:
                            mask[y, x] = 255
                    else:
                        if theta >= angle_start or theta < angle_end:
                            mask[y, x] = 255
        return cv2.bitwise_and(self.img, self.img, mask=mask)


    def obtener_sectores(self) -> Dict[str, np.ndarray]:
        """
        Devuelve un diccionario con los nombres de los sectores y sus imágenes correspondientes.
        Listo para ser usado como input para un modelo YOLO o similar.
        """
        return self.sectores

    
    def mostrar_sectores(self):
        fig, axs = plt.subplots(1, 4, figsize=(16, 6))
        for ax, (nombre, sector_img) in zip(axs, self.sectores.items()):
            img_rgb = cv2.cvtColor(sector_img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(nombre)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    

    def dibujar_rectangulos_rotados_en_sectores(self, rectangulos_por_sector: dict):
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        direcciones = ['Norte', 'Sur', 'Este', 'Oeste']

        for idx, direccion in enumerate(direcciones):
            clave = direccion
            img = self.sectores[clave].copy()

            # Obtener los rectángulos para esta dirección (ahora es un diccionario con claves numéricas)
            rect_dict = rectangulos_por_sector.get(clave, {})
            
            # Iterar a través de los elementos del diccionario
            for num, rect in rect_dict.items():
                cx, cy = rect["cx"], rect["cy"]
                w, h = rect["w"], rect["h"]
                angle = rect["angle"]

                # Define el rectángulo rotado (formato: centro, tamaño, ángulo)
                rect_cv = ((cx, cy), (w, h), angle)

                # Obtiene los 4 puntos del rectángulo
                box = cv2.boxPoints(rect_cv)
                box = np.intp(box)

                # Dibuja el rectángulo verde
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                
                # Dibuja el número del rectángulo en lugar del círculo rojo
                # Parámetros de texto: imagen, texto, posición, fuente, escala, color, grosor
                cv2.putText(
                    img=img,
                    text=str(num),
                    org=(int(cx), int(cy)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(0, 0, 255),  # Rojo
                    thickness=2
                )
                
            axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[idx].set_title(direccion)
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()