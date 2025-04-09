# 🚗 Parking360

**Detector de celdas vacías en parqueaderos usando cámara 360°.**

Este proyecto utiliza visión por computador y modelos de detección para identificar espacios libres en un parqueadero capturado con una cámara 360.

---

## 📦 Requisitos

- Python 3.12 o superior
- Git (opcional, si clonas el proyecto)

---

## 🚀 Ejecutar Parking360 con Docker

Este proyecto usa [`uv`](https://docs.astral.sh/uv) para gestionar dependencias, y Docker para entorno aislado.

### 🐳 Requisitos
- Docker instalado (`>= 20.10`)
- (Opcional) Acceso a cámara: `--device=/dev/video0`

### 📦 Build de la imagen


-git clone https://github.com/soric91/parkiu_device.git

-cd parkiu_device

-docker build -t parking360 .

-docker run --rm -it parking360
