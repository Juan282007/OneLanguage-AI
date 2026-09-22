# Cambios realizados - OneLanguage

Este documento resume los cambios aplicados al modelo de Lengua de Senas Colombiana (LSC) y su integracion con los proyectos frontend y backend.

## Inventario por repositorio

### OneLanguage-AI

- Se agrego `web_service.py` para publicar el modelo por WebSocket.
- Se actualizaron `requirements.txt` y `README.md` con la ejecucion del servicio web.
- Se documentaron las mejoras de preprocesamiento, reconocimiento temporal y clase neutral que ya estaban siendo trabajadas en `lsc_pipeline.py`, `recognizer.py` y `train_from_videos.py`.

### OneLanguage_Frontend/web

- `src/pages/translate/Translate.jsx` y `Translate.css`: camara, WebSocket de IA, texto, confianza y voz del navegador.
- `src/pages/login/Login.jsx`: correccion de la ruta de recuperacion y redireccion inmediata a `/home` despues de un login correcto.
- `src/services/api.js`: limpieza de espacios en `VITE_API_URL`.
- `.env.example`: variables correctas de Vite para API e IA.
- `.env` local: se retiro el espacio inicial de la URL configurada. Este archivo esta ignorado por Git y no contiene cambios para subir.

### OneLanguage-Backend

- `docker-compose.yml`: se pasa `APP_FRONTEND_BASE_URL` al contenedor.
- `Backend-OneLanguage/src/main/resources/application.properties`: propiedad configurable `app.frontend-base-url`.
- `PasswordResetServiceImpl.java`: el enlace enviado por correo deja de usar una URL fija de localhost y usa la configuracion anterior.
- `.env.example`: documentacion de `APP_FRONTEND_BASE_URL`.

### Base de datos

No se modifico la estructura ni los datos de PostgreSQL. No se ejecutaron migraciones, `INSERT`, `UPDATE`, `DELETE`, `DROP` ni cambios de Liquibase. Solo se verifico que la base estaba saludable, que el backend podia conectarse y que existia un usuario con hash BCrypt valido.

## Modelo de IA LSC

- `lsc_pipeline.py` centraliza la extraccion de landmarks de MediaPipe, normalizacion, suavizado, remuestreo y preparacion de secuencias.
- Cada entrada usa una secuencia temporal fija de 32 fotogramas, hasta dos manos y caracteristicas de movimiento.
- `recognizer.py` usa una ventana temporal, umbral de confianza, margen entre las mejores clases y varias predicciones consecutivas antes de traducir.
- Los nombres seguros de carpetas se convierten en texto para las personas: `buenos-dias` se muestra como `buenos dias`.

## Clase neutral `__idle__`

Se incorporo la etiqueta tecnica `__idle__` para videos sin una sena valida. Sirve para reducir traducciones falsas por manos en reposo, gestos casuales o movimientos incompletos.

- Sus videos se guardan en `videos/__idle__/`.
- No se muestra como texto ni se reproduce por voz.
- Los clips neutrales se dividen en ventanas densas durante el entrenamiento.
- Al predecir `__idle__`, la inferencia limpia su ventana temporal y se mantiene en espera.

Limitacion conocida: `idle_IMG_0724.MOV` se clasifico como `buenos-dias` con alta confianza. Es un problema de datos y del modelo entrenado, no del WebSocket ni del frontend. Se debe agregar ese tipo de video dificil a `videos/__idle__/`, reentrenar y evaluar de nuevo.

## Servicio web del modelo

Se agrego `web_service.py` para exponer el modelo al navegador sin mover TensorFlow o MediaPipe al frontend.

- `GET /health`: confirma la carga del modelo.
- `GET /model-info`: devuelve metadatos y etiquetas activas.
- `WS /ws/recognize`: recibe JPEG desde la camara y devuelve estado, texto y confianza.

Cada navegador tiene una sesion independiente y una ventana propia de landmarks. React no tiene etiquetas quemadas: las recibe desde `model/lsc_labels.json`.

```powershell
cd C:\Users\juanp\Desktop\OneLanguage-AI
.\venv\Scripts\Activate.ps1
python.exe web_service.py
```

El servicio usa `http://localhost:8000` y `ws://localhost:8000/ws/recognize`.

## Integracion con frontend

La pantalla de traduccion fue actualizada para:

- Solicitar y mostrar la camara del navegador.
- Enviar JPEG reducidos al servicio cada 100 ms, sin acumular solicitudes.
- Mostrar estado, traduccion y confianza.
- Reproducir la nueva traduccion estable mediante la voz del navegador en espanol colombiano.
- Cerrar camara, WebSocket y voz al finalizar.

Variables de entorno del frontend:

```env
VITE_API_URL=http://localhost:8084
VITE_AI_WS_URL=ws://localhost:8000/ws/recognize
```

Para agregar senas nuevas no se modifica React: se reentrena, se reemplazan juntos `lsc_sequence_model.keras`, `lsc_labels.json` y `lsc_config.json` en `model/`, y se reinicia `web_service.py`.

## Autenticacion y recuperacion

- Se verifico que PostgreSQL y Spring Boot inician y se conectan correctamente.
- Se corrigio la ruta de recuperacion del frontend a `/forgotpassword`.
- Un login correcto guarda el JWT y redirige inmediatamente a `/home`.
- `VITE_API_URL` se normaliza con `trim()` para evitar espacios accidentales.
- Se corrigio el `.env.example` del frontend, que tenia variables del backend.
- La URL de recuperacion ya no esta fija en Java: el backend usa `APP_FRONTEND_BASE_URL`.

En el `.env` local del backend debe definirse la URL desde la cual se abre el frontend. Ejemplo para red local:

```env
APP_FRONTEND_BASE_URL=http://192.168.1.17:5173
```

Despues se reinicia el backend:

```powershell
cd C:\Users\juanp\Desktop\OneLanguage-Backend
docker compose up -d --build
```

Las credenciales de Gmail permanecen fuera del repositorio. Si no llega un correo, revisar la contrasena de aplicacion de Gmail, el remitente y la carpeta de spam.

## Validaciones ejecutadas

- Carga del modelo mediante `GET /health`.
- Conexion WebSocket y envio de fotogramas JPEG al servicio de IA.
- `npm.cmd run lint` y `npm.cmd run build` del frontend.
- Build y arranque de Docker del backend en el puerto `8084`.
- Respuesta `401` para login invalido y `400` al solicitar recuperacion para un correo no registrado.

## Proximos pasos

1. Agregar mas videos de cada sena y de `__idle__`, con ambas manos, distintos fondos y velocidades.
2. Reentrenar y evaluar con videos que no participaron en el entrenamiento.
3. Configurar correo de produccion con contrasena de aplicacion.
4. Publicar IA con HTTPS/WSS y definir dominios reales para `ALLOWED_ORIGINS`, `VITE_API_URL`, `VITE_AI_WS_URL` y `APP_FRONTEND_BASE_URL`.
