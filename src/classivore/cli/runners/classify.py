#!/usr/bin/env python3
"""Runner for `classivore classify`.

Heavy imports (torch via Classifier) live inside run() so that other
commands don't pay the load cost.
"""

import sys


def run(args):
    import json
    from pathlib import Path

    from classivore.config.settings import get_models_dir
    from classivore.inference import Classifier

    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        models_dir = get_models_dir()
        slug_dir = models_dir / args.taxonomy
        if not slug_dir.is_dir():
            print(f"Error: no models found for taxonomy '{args.taxonomy}' in {models_dir}")
            sys.exit(1)
        subdirs = sorted(d for d in slug_dir.iterdir() if d.is_dir() and d.name != "checkpoints")
        if not subdirs:
            print(f"Error: no model runs found in {slug_dir}")
            sys.exit(1)
        model_dir = subdirs[-1]

    print(f"Loading model from {model_dir}...")
    try:
        classifier = Classifier(model_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.text:
        results = classifier.predict(args.text)
        if args.output:
            with open(args.output, "w") as f:
                json.dump({"text": args.text, "predictions": results}, f, indent=2)
            print(f"Results written to {args.output}")
        else:
            _print_results(results)

    elif args.file:
        input_path = Path(args.file)
        if not input_path.exists():
            print(f"Error: file not found: {input_path}")
            sys.exit(1)

        raw = input_path.read_text().strip()
        if raw.startswith("["):
            records = json.loads(raw)
        else:
            records = [json.loads(line) for line in raw.split("\n") if line.strip()]

        texts = [r["text"] for r in records]
        all_results = classifier.predict_batch(texts)

        output_records = []
        for record, preds in zip(records, all_results):
            out = {k: v for k, v in record.items() if k != "text"}
            out["text"] = record["text"]
            out["predictions"] = preds
            output_records.append(out)

        output_lines = [json.dumps(r) for r in output_records]
        output_text = "\n".join(output_lines) + "\n"
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_text)
            print(f"Classified {len(records)} texts → {args.output}")
        else:
            print(output_text, end="")

    elif args.interactive:
        print("Classivore interactive mode. Enter text, press Enter. Ctrl-D to exit.")
        try:
            while True:
                text = input("> ")
                if not text.strip():
                    continue
                results = classifier.predict(text)
                _print_results(results)
        except EOFError:
            print()

    else:
        print("Error: specify --text, --file, or --interactive")
        sys.exit(1)


def _print_results(results):
    """Print classification results as a formatted table."""
    if not results:
        print("  No categories above threshold.")
        return

    name_width = max(len(r["name"]) for r in results)
    name_width = max(name_width, 8)

    line = "─" * (name_width + 40)
    print(f"\n  {'Category'.ljust(name_width)}  {'Confidence':>10}  Path")
    print(f"  {line}")
    for r in results:
        path_str = " > ".join(r["path"]) if r.get("path") else ""
        print(f"  {r['name'].ljust(name_width)}  {r['confidence']:>10.4f}  {path_str}")
    print()
