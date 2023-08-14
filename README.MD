

# Instrucciones para ejecutar el programa

Este archivo README proporciona información sobre cómo ejecutar el programa utilizando el archivo `main.py`. El programa está diseñado para procesar videos utilizando la biblioteca Python y OpenCV. A continuación, se describen las opciones de ejecución y cómo proporcionar la entrada adecuada al programa.

## Uso del programa

El programa puede ser ejecutado desde la línea de comandos con diferentes opciones. A continuación, se explica cómo usar estas opciones:

```bash
python main.py --source <un-video>
```

- Si proporciona la opción `--source` seguida de la ruta de un archivo de video, el programa procesará ese video en particular. Por ejemplo:

```bash
python main.py --source path/to/video.mp4
```

- Si no proporciona la opción `--source`, el programa utilizará la cámara por defecto de su dispositivo para capturar el video en tiempo real. Por ejemplo:

```bash
python main.py
```

## Ejemplos

1. Procesar un video específico:
```bash
python main.py --source path/to/video.mp4
```

2. Usar la cámara por defecto:
```bash
python main.py
```

## Notas

- Asegúrese de proporcionar la ruta correcta al archivo de video si está utilizando la opción `--source`.

- El programa utiliza OpenCV para manejar la captura y procesamiento de video. Puede revisar la documentación de OpenCV para obtener más información sobre cómo realizar ajustes avanzados: https://docs.opencv.org/

- Si encuentra algún problema o tiene preguntas, no dude en ponerse en contacto con el equipo de desarrollo.

¡Disfrute usando el programa para procesar videos o capturar desde su cámara!