# Cambios De Frontend E IA

Este documento resume los cambios aplicados en las ramas `HU-18-dev` de OneLanguage_Frontend y `feature/lsc-new-signs` de OneLanguage-AI. No se realizaron cambios en backend ni base de datos.

## Frontend

- Se integró un aviso previo de permiso de cámara desde Inicio, tanto en web como en móvil.
- Al aceptar, web solicita acceso con `getUserMedia` y móvil abre el permiso nativo de Expo Camera antes de entrar al traductor.
- Al rechazar o denegar el permiso, el usuario permanece en Inicio y recibe un mensaje claro.
- Se añadieron textos del aviso en español, inglés, portugués e italiano.
- Se conservaron las mejoras de integración del traductor: flujo de cámara, mensajes de estado, WebSocket de IA y manejo de sesión.

## Modelo De IA

- Se mantuvo un modelo secuencial de landmarks de MediaPipe para traducir señas a texto.
- Se retiró la clase neutra del entrenamiento activo; las carpetas de datos y modelos generados no se suben al repositorio.
- Se añadió normalización lateral para señas de una mano: izquierda y derecha llegan al modelo con la misma orientación y posición de entrada.
- Se conservó el orden normal de landmarks para señas de dos manos.
- Se amplió la tolerancia a pérdidas breves de detección durante giros de palma, evitando reiniciar una seña continua demasiado pronto.
- Se mantuvo la confirmación temporal, el umbral de confianza y el margen entre clases para reducir traducciones inestables.
- Se añadió `evaluate_live.py` para medir videos de forma similar a una cámara a 5 FPS.
- Se actualizó `.gitignore` para excluir `dataset_v2/`, además de videos, evaluaciones, modelos, entornos virtuales y secretos locales.

## Operación

Después de entrenar un modelo nuevo, reinicie el servicio para cargarlo en web y móvil:

```powershell
docker compose up --build -d
Invoke-RestMethod http://localhost:8001/model-info
```

El endpoint debe mostrar las etiquetas y la versión de preprocesamiento del modelo recién entrenado.
