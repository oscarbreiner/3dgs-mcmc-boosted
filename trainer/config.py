import json


def load_config(config_file):
    with open(config_file, "r") as file:
        return json.load(file)


def get_cli_keys(argv):
    keys = set()
    for token in argv:
        if token.startswith("--"):
            key = token[2:].split("=")[0]
            if key:
                keys.add(key)
        elif token.startswith("-") and len(token) > 1:
            key = token[1:2]
            if key:
                keys.add(key)
    return keys


def apply_config_overrides(args, argv):
    config = {}
    cli_keys = get_cli_keys(argv)
    if getattr(args, "config", None) is not None:
        config = load_config(args.config)
        for key, value in config.items():
            if key not in cli_keys:
                setattr(args, key, value)
    return args, config, cli_keys


def apply_default_test_iterations(args, cli_keys, config):
    if "test_iterations" not in cli_keys and "test_iterations" not in config:
        args.test_iterations = list(range(0, int(args.iterations) + 1, 5_000))
        if not args.test_iterations or args.test_iterations[-1] != int(args.iterations):
            args.test_iterations.append(int(args.iterations))
