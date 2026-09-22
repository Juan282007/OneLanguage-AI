# One Language - Traductor de Lengua de Senas Colombiana (LSC)

Modelo local de inteligencia artificial para reconocer señas aisladas de Lengua de Senas Colombiana (LSC) a partir de una camara web o de un archivo de video. Muestra la traduccion como texto y puede leerla en voz alta.

El proyecto esta pensado como el componente de vision de **One Language**. Se entrena con videos propios organizados por sena, por lo que las palabras que reconoce dependen directamente de las carpetas disponibles en el dataset.

> Alcance actual: clasificacion de señas aisladas. Todavia no traduce conversaciones continuas ni frases completas dentro de un video largo. Para reconocer una nueva sena debe existir una carpeta con ejemplos de esa sena y se debe volver a entrenar el modelo.

## Como funciona

1. La camara o el video entrega fotogramas al programa.
2. MediaPipe localiza hasta dos manos y extrae 21 puntos por mano (muneca y dedos).
3. El proyecto convierte esos puntos en una secuencia de 32 fotogramas. Conserva la forma de cada mano y la trayectoria de las munecas para distinguir el movimiento.
4. Un modelo TensorFlow con capas BiLSTM analiza la secuencia y devuelve la clase mas probable, por ejemplo `hola`, `bien` o `mal`.
5. El reconocedor exige confianza y separacion suficiente frente a la segunda opcion antes de confirmar el texto. Esto reduce traducciones falsas cuando una sena se parece a otra.
6. Al confirmar una clase, la muestra en pantalla y, si la voz esta activa, `pyttsx3` la pronuncia.

Durante el entrenamiento se generan versiones espejo de las secuencias para que el modelo aprenda la sena realizada con la mano contraria. Aun asi, es importante conservar algunos videos reales hechos con ambas manos para evaluar el resultado.

## Estructura del proyecto

```text
AI-model/
  main.py                 Punto de entrada para camara o video.
  recognizer.py           Interfaz de reconocimiento en tiempo real y voz.
  lsc_pipeline.py         Extraccion de manos, secuencias y metadatos.
  train_from_videos.py    Entrenamiento desde las carpetas de videos.
  requirements.txt        Dependencias de Python.
  videos/                 Dataset local, una carpeta por etiqueta. No se sube a Git.
  model/                  Modelo y etiquetas generados al entrenar. No se sube a Git.
  README.md               Esta guia.
```

## Dataset: carpeta `videos`

Cada subcarpeta de `videos` representa una etiqueta. Su nombre sera exactamente el texto que aparezca como traduccion.

```text
videos/
  hola/
    video1.mp4
    video2.mp4
  bien/
    video1.mp4
  mal/
    video1.mp4
  regular/
    video1.mp4
  __idle__/
    video1.mp4
```

### Que guardar en cada carpeta

- `hola/`, `bien/`, `mal/`, `regular/`: un video por archivo con una unica sena completa.
- `__idle__/`: ejemplos de ausencia de sena. Incluye manos relajadas, transiciones entre senas, manos entrando o saliendo del encuadre, movimientos cotidianos que no son LSC y momentos de espera. No debe contener ninguna sena LSC valida o incompleta.

`__idle__` es una clase tecnica de no-traduccion: nunca se muestra como palabra ni se reproduce por audio. Se evita el nombre `nada` porque este puede reservarse para la sena LSC que expresa esa palabra. Durante la carga, alias como `no_sign`, `neutral`, `idle` o `reposo` se normalizan internamente a `__idle__`.

El entrenamiento cubre `__idle__` con ventanas densas a lo largo de todo el video. Asi aprende tambien las fases intermedias de un movimiento cotidiano que la ventana deslizante de la camara observara en tiempo real. Los pesos de clase compensan automaticamente la mayor cantidad de ventanas negativas.

### Recomendaciones para grabar

- Usa clips de 1 a 2 segundos: reposo breve, una sena completa y reposo breve.
- Para `__idle__`, usa clips de 2 a 5 segundos que contengan solo una situacion sin sena.
- Manten ambas manos completas dentro del encuadre, con buena luz y fondo con contraste.
- Graba entre 15 y 30 ejemplos por sena como base; mezcla distancias, velocidad, ropa, fondos y, cuando sea posible, distintas personas.
- Repite una cantidad parecida de videos por etiqueta. Una clase con muchos mas ejemplos puede dominar las predicciones.
- Si `regular` se confunde con `mal`, revisa y vuelve a grabar ejemplos de ambas, no solo de una. Los videos deben mostrar claramente la diferencia de forma, direccion y recorrido.
- Si el entrenamiento indica que no detecto manos en un clip, ese archivo no ayuda al modelo: vuelve a grabarlo con las manos mas visibles.

Los videos son datos de entrenamiento locales. Pueden contener imagenes de personas, por lo que no se incluyen en el repositorio por privacidad, tamano y derechos de uso.

## Instalacion desde un repositorio clonado

Requiere Python 3.11, una camara web para el modo en vivo y Windows para la salida de voz actual.

```powershell
git clone <URL_DEL_REPOSITORIO>
cd AI-model
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activacion del entorno virtual, ejecuta una vez en esa misma terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Tambien se puede evitar la activacion ejecutando directamente `venv\Scripts\python.exe` en los comandos siguientes.

Como `videos/` y `model/` no se versionan, al clonar se debe hacer una de estas dos cosas:

1. Crear o copiar el dataset local en `videos/` y entrenar el modelo.
2. Descargar un modelo publicado como artefacto de una version del proyecto y colocarlo en `model/` junto con sus archivos de etiquetas y configuracion.

## Entrenar el modelo

Con el entorno virtual activo y el dataset listo:

```powershell
python train_from_videos.py --data videos --epochs 100 --augmentations 3 --windows-per-video 3
```

El entrenamiento crea estos archivos:

```text
model/
  lsc_sequence_model.keras  Modelo TensorFlow entrenado.
  lsc_labels.json           Etiquetas y orden de salida del modelo.
  lsc_config.json           Configuracion de preprocesamiento compatible.
```

Vuelve a entrenar cada vez que agregues, elimines o sustituyas videos, o despues de cambiar la forma en que se extraen las secuencias. No mezcles archivos de modelo de una ejecucion con etiquetas o configuraciones de otra.

Al terminar, revisa en la terminal los bloques `Validacion por sena` y `Validacion espejo`. Una precision alta con pocos clips no garantiza buen uso real: prueba tambien videos nuevos que no hayan participado en el entrenamiento.

## Ejecutar la traduccion

### Servicio para frontend web

El frontend web usa el servicio WebSocket de este repositorio para enviar frames de la camara y recibir texto en tiempo real. Los videos nunca salen del navegador ni se guardan: cada frame se procesa en memoria y se descarta.

Con el paquete compatible dentro de `model/`, inicia el servicio:

```powershell
python -m pip install -r requirements.txt
python web_service.py
```

El servicio queda disponible en `http://localhost:8000` y expone `GET /health`, `GET /model-info` y `WS /ws/recognize`.

Para permitir otro origen, configura `ALLOWED_ORIGINS` antes de iniciarlo. Ejemplo: `ALLOWED_ORIGINS=http://localhost:5173`.

Para desarrollo en este equipo, abre el frontend en `http://localhost:5173`; los navegadores permiten usar la camara en `localhost`. Si se abre mediante una IP como `http://192.168.x.x:5173`, la camara sera bloqueada por el navegador porque esa direccion no usa HTTPS. Para pruebas desde otra maquina o un telefono, publica el frontend con HTTPS y agrega su origen `https://...` a `ALLOWED_ORIGINS` en un archivo `.env` basado en `.env.example`.

Cuando se entrenen nuevas senas, publica juntos `lsc_sequence_model.keras`, `lsc_labels.json` y `lsc_config.json` dentro de `model/` y reinicia el servicio. El frontend recibe las etiquetas desde el servicio, por lo que no requiere una lista de senas quemada en codigo.

### Camara web

```powershell
python main.py --fullscreen
```

Sin `--fullscreen`, la ventana inicia en 1280x720. Se pueden ajustar sus dimensiones:

```powershell
python main.py --width 1280 --height 720
```

### Video guardado

```powershell
python main.py --source ruta\al\video.mp4
```

### Opciones utiles

```powershell
python main.py --threshold 0.70 --margin 0.12
python main.py --no-speak
python main.py --no-auto-speak
```

- `--threshold`: confianza minima para aceptar una traduccion. Si se omite, se usa el valor guardado durante el entrenamiento.
- `--margin`: diferencia minima entre la mejor prediccion y la segunda mejor. Aumentarlo reduce falsos positivos, pero puede tardar mas en confirmar una sena.
- `--stable-predictions`: cantidad de predicciones consecutivas iguales necesarias para confirmar una traduccion.
- `--no-speak`: inicia sin voz.
- `--no-auto-speak`: muestra texto sin pronunciarlo automaticamente.

## Controles en la ventana

| Tecla | Accion |
| --- | --- |
| `ESC` | Cerrar la camara o video. |
| `V` | Activar o desactivar la voz. |
| `A` | Activar o desactivar la lectura automatica. |
| `F` | Alternar pantalla completa. |
| `ESPACIO` | Pronunciar la traduccion estable actual. |

## Diagnostico rapido

| Situacion | Que hacer |
| --- | --- |
| Todas las senas se convierten en `hola` | Agrega ejemplos variados de las otras clases, crea `__idle__/`, revisa videos mal etiquetados y reentrena. |
| Una mano funciona y la otra se confunde | Entrena de nuevo con la version actual y agrega algunos ejemplos reales con la mano contraria. |
| `mal` y `regular` se confunden | Revisa los clips de ambas clases, graba diferencias mas claras y equilibra la cantidad de videos. |
| No aparece una traduccion | Mejora la luz, aleja un poco las manos para que se vean completas o baja cuidadosamente `--threshold`/`--margin`. |
| No hay audio | Pulsa `V`, verifica que el audio de Windows funcione y confirma que `pyttsx3` este instalado desde `requirements.txt`. |
| Dice que no existe el modelo | Ejecuta primero el comando de entrenamiento. |
| Indica que el modelo es anterior | Reentrena para generar `lsc_config.json` y usar el preprocesamiento actual. |

## Flujo de mejora recomendado

1. Graba ejemplos limpios y balanceados de cada sena e incluye negativos variados en `__idle__`.
2. Entrena y conserva la salida de validacion.
3. Prueba la camara con clips nuevos y con ambas manos.
4. Guarda los errores por pareja de senas, por ejemplo `mal` -> `hola`.
5. Corrige clips ambiguos y agrega ejemplos que representen esos errores.
6. Reentrena y compara resultados antes de integrar una nueva version al backend o a las apps.

Para una aplicacion movil o web, no se envian las carpetas de entrenamiento al usuario final. Se distribuye un modelo optimizado y versionado como artefacto, junto con sus etiquetas y configuracion compatibles. El dataset se conserva de forma privada para mejorar futuras versiones.

## Git y archivos locales

El archivo `.gitignore` excluye el entorno virtual, los videos, los modelos generados, caches, bases de datos locales y configuraciones con posibles secretos. El repositorio debe contener el codigo, las dependencias y esta guia; los datos y modelos se manejan por separado como artefactos privados o publicados de forma controlada.
