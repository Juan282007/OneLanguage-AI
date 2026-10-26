from recognizer import SignRecognizer, parse_args

if __name__ == "__main__":
    args = parse_args()
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
    app.run()
