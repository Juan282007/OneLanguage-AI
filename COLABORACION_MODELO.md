# Colaboracion y Distribucion del Modelo LSC

Esta guia explica como trabajar entre varias personas sin subir los videos de entrenamiento al repositorio. El proyecto separa el codigo, el modelo ya entrenado y el dataset privado.

## Que contiene cada componente

| Componente | Contenido | Donde se comparte |
| --- | --- | --- |
| Repositorio Git | Codigo Python, dependencias y documentacion | GitHub, GitLab o Azure DevOps |
| Paquete de modelo | Pesos entrenados, etiquetas y configuracion | Release privado o almacenamiento privado |
| Dataset | Videos originales organizados por clase | Almacenamiento privado con acceso controlado |

Los videos no deben subirse al repositorio normal: son pesados, pueden contener datos personales y cambian con frecuencia.

## Escenario 1: ejecutar el traductor

Este escenario aplica cuando un companero solo necesita probar la camara, integrar el modelo o trabajar en frontend y backend. No necesita los videos.

### Paso 1. Clonar el repositorio

```powershell
git clone <URL_DEL_REPOSITORIO>
cd OneLanguage-AI
```

### Paso 2. Crear el entorno e instalar dependencias

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activacion:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### Paso 3. Descargar el paquete de modelo

El responsable de IA publica un archivo versionado, por ejemplo:

```text
one-language-lsc-model-v1.1.0.zip
```

Al extraerlo en la raiz del proyecto, la estructura debe quedar asi:

```text
OneLanguage-AI/
  model/
    lsc_sequence_model.keras
    lsc_labels.json
    lsc_config.json
```

Los tres archivos son obligatorios y deben provenir de la misma entrega.

- `lsc_sequence_model.keras`: contiene lo aprendido por TensorFlow.
- `lsc_labels.json`: relaciona cada salida del modelo con una seña, incluido `__idle__`.
- `lsc_config.json`: define la longitud de secuencia, cantidad de caracteristicas y parametros compatibles.

No se debe reemplazar solo el archivo `.keras`: mezclarlo con etiquetas o configuraciones de otra version genera resultados incorrectos o un error de compatibilidad.

### Paso 4. Ejecutar la camara

```powershell
python.exe main.py --source 1 --fullscreen
```

Si la camara principal corresponde a otro indice, cambia `1` por `0`, `2` u otro numero disponible.

## Escenario 2: agregar senas o reentrenar

Quien vaya a mejorar el modelo necesita el paquete anterior y ademas el dataset privado. El modelo entrenado no reemplaza los videos: sin dataset no se pueden validar ni entrenar nuevas versiones de manera confiable.

### Paso 1. Obtener el dataset privado

El dataset se comparte por OneDrive, Google Drive, Dropbox, S3 u otro almacenamiento privado. Debe conservar las carpetas por etiqueta:

```text
videos/
  __idle__/
    idle_001.mov
  hola/
    video1.mp4
  bien/
    video1.mp4
  buenas-tardes/
    video1.mp4
  mal/
    video1.mp4
```

Cada carpeta representa una clase. Los nombres con guion, como `buenas-tardes`, se muestran como `buenas tardes` al usuario.

`__idle__` es una clase tecnica: representa manos en reposo, movimientos cotidianos, pausas y acciones que no son LSC. No se muestra ni se pronuncia como traduccion. Se reserva el nombre `nada` para una posible seña LSC real en el futuro.

### Paso 2. Agregar videos correctamente

1. Crear o usar la carpeta de la seña correspondiente.
2. Guardar un video por archivo y una seña principal por video.
3. Mantener cantidades parecidas por clase.
4. Grabar tambien con mano izquierda, derecha y ambas cuando la seña lo permita.
5. Revisar que no existan videos vacios, corruptos o mal etiquetados.
6. Guardar pruebas nuevas fuera de `videos/` para no entrenar y evaluar con el mismo material.

No crear carpetas auxiliares como `__MACOSX`, `evaluacion` o `rechazados` dentro de `videos/`, porque pueden ser interpretadas como clases. El entrenamiento ignora `__MACOSX`, pero las pruebas deben mantenerse fuera del dataset.

### Paso 3. Entrenar

```powershell
python.exe train_from_videos.py --data videos --epochs 100 --augmentations 3 --windows-per-video 3
```

El entrenamiento realiza lo siguiente:

1. Corrige la orientacion de videos moviles cuando existe metadato de rotacion.
2. Extrae landmarks de hasta dos manos con MediaPipe.
3. Normaliza los movimientos a secuencias de 32 frames.
4. Entrena un modelo BiLSTM temporal con las senas y `__idle__`.
5. Usa pesos de clase para compensar cantidades diferentes de videos o ventanas.
6. Guarda el nuevo modelo, etiquetas y configuracion en `model/`.

Al finalizar, revisar los bloques `Validacion por ventanas de cada sena` y `Validacion espejo`. Una precision alta no sustituye probar con personas, fondos y grabaciones que no participaron en el entrenamiento.

## Publicar una nueva version del modelo

Cada entrenamiento aprobado debe convertirse en un paquete nuevo. Ejemplo:

```text
one-language-lsc-model-v1.2.0.zip
  model/
    lsc_sequence_model.keras
    lsc_labels.json
    lsc_config.json
```

Recomendacion de versiones:

| Cambio | Ejemplo |
| --- | --- |
| Correccion menor o nuevo entrenamiento sin clases nuevas | `v1.1.1` |
| Nueva seña o nueva clase `__idle__` | `v1.2.0` |
| Cambio incompatible de preprocesamiento | `v2.0.0` |

Junto al ZIP, publicar una nota corta con la version del dataset, clases incluidas, fecha, metricas de validacion y cambios conocidos.

## Donde alojar los paquetes

Opciones recomendadas:

- GitHub Releases privados para paquetes de modelo pequenos o medianos.
- OneDrive, Google Drive, Dropbox, S3 o Azure Blob para modelos y datasets privados.
- Git LFS solo cuando el equipo ya lo administre y acepte sus cuotas de almacenamiento y transferencia.

No usar el repositorio Git normal para archivos de video. Tampoco compartir el dataset mediante enlaces publicos si contiene personas sin consentimiento explicito.

## Flujo recomendado del equipo

```text
Videos privados
  -> revision de etiqueta y calidad
  -> dataset versionado
  -> entrenamiento y pruebas
  -> paquete de modelo versionado
  -> release privado
  -> aplicacion web, Android, iOS o backend
```

La aplicacion final descarga o incorpora solo el paquete de modelo. Los videos permanecen privados y se usan unicamente para mejorar futuras versiones.

## Lista de verificacion antes de compartir

- El paquete contiene los tres archivos dentro de `model/`.
- Las etiquetas y la configuracion pertenecen al mismo modelo `.keras`.
- Se probo `__idle__` para confirmar que no haya traducciones durante movimientos cotidianos.
- Se probaron senas reales con ambas manos cuando corresponda.
- El dataset sigue excluido por `.gitignore`.
- Los respaldos del modelo anterior estan disponibles antes de publicar una nueva version.
