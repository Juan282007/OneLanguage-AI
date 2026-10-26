from recognizer import SignRecognizer, parse_args

# BLOQUE: punto de entrada de la aplicacion de escritorio.
# Este archivo no contiene la inteligencia del modelo. Su funcion es unir las
# opciones escritas en consola con `SignRecognizer`, que carga el paquete IA,
# abre la camara o video y mantiene el ciclo de reconocimiento.
if __name__ == "__main__":
    # Lee fuente de video, dimensiones, voz y umbrales sin dejar valores fijos
    # en el codigo. Asi la misma aplicacion sirve para camara y archivos locales.
    args = parse_args()
    # Se inyectan ajustes de interfaz e inferencia; no se cambian los pesos ni
    # las etiquetas del modelo, solo la forma en que la sesion los utiliza.
    app = SignRecognizer(
        source=args.source,
        speak=not args.no_speak,
        confidence_threshold=args.threshold,
        width=args.width,
        height=args.height,
        fullscreen=args.fullscreen,
        auto_speak=not args.no_auto_speak,
        minimum_margin=args.margin,
    )
    # Inicia el ciclo que lee frames, prepara secuencias, predice, muestra texto
    # y libera recursos al cerrar la ventana.
    app.run()
