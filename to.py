#!/usr/bin/env python3.11
# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Optional, Union

try:
    import argcomplete
except ModuleNotFoundError:
    pass

Format = Literal["json", "json5", "toml", "yaml", "python"]
SUPPORTED_FORMATS: list[Format] = Format.__args__
SUPPORTED_FORMATS_STR: str = ", ".join(SUPPORTED_FORMATS)
DIFF_OUTPUT_TOOLS_HELP = {
    "A file path": "Save the diff output to this file",
    "diff": "Show diff in terminal using coreutils diff",
    "delta": "Open in delta diff viewer",
    "code": "Open diff in VS Code",
    "pycharm": "Open diff in PyCharm",
}

DIFF_OUTPUT_TOOLS_HELP_FORMATTED = "\n".join(
    f"- {k!r}: {v}" for k, v in DIFF_OUTPUT_TOOLS_HELP.items()
)
STDOUT = "stdout"
SETTINGS = {"max_width": 120, "sort_keys": True}


# ===[ Functional ]===
# ---[ Helpers ]---
def stderr(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)


def cache[T, **P](func: Callable[P, T]) -> Callable[P, T]:
    cache_dict = {}

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        key = args + tuple(sorted(kwargs.items()))
        if key in cache_dict:
            return cache_dict[key]
        result = func(*args, **kwargs)
        cache_dict[key] = result
        return result

    return wrapper


# region ---[ Imports] ---
def import_json5():
    try:
        import json5
    except ImportError:
        stderr("json5 module not found. Please install via 'pip install json5'")
        sys.exit(1)
    return json5


def import_toml():
    try:
        import toml
    except ImportError:
        stderr("toml module not found. Please install via 'pip install toml'")
        sys.exit(1)
    return toml


def import_yaml():
    try:
        from ruamel.yaml import YAML
    except ImportError:
        stderr(
            "ruamel.yaml module not found. Please install via 'pip install \"ruamel.yaml\"'"
        )
        raise

    def str_representer(dumper, data):
        if len(data.splitlines()) > 1:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml = YAML(typ="safe", pure=True)

    # Not supported with typ="safe", pure=True.
    # typ="safe", pure=False was flaky with applying '|'.
    # yaml.comment_handling = "preserve"
    yaml.default_flow_style = False
    yaml.indent = 4
    yaml.preserve_quotes = True
    yaml.representer.add_representer(str, str_representer)
    yaml.sequence_dash_offset = 2
    yaml.sort_base_mapping_type_on_output = True
    yaml.sort_keys = SETTINGS["sort_keys"]
    yaml.width = 999
    return yaml


def import_rich():
    try:
        import rich
    except ImportError:
        stderr(
            "rich module not found. It is required when specifying -p or --pretty. Please install via 'pip install rich'"
        )
        sys.exit(1)
    return rich


def import_pyperclip():
    try:
        import pyperclip
    except ImportError:
        stderr("pyperclip module not found. Please install via 'pip install pyperclip'")
        sys.exit(1)
    return pyperclip


# endregion


# region ---[ I/O Availability Checks ]---
def is_piped() -> bool:
    try:
        # Note: sgpt does stdin_passed=not stdin_available()
        return not stdin_available() and stdin_has_value()
    except AttributeError:
        return False


@cache
def stdin_available() -> bool:
    try:
        return sys.stdin.isatty()
    except AttributeError:
        return False


@cache
def stdin_has_value() -> bool:
    return bool(read_stdin())


@cache
def read_stdin() -> str:
    """stdin can be read only once, so cache the result."""
    return sys.stdin.read()


@cache
def stdout_available() -> bool:
    try:
        return sys.stdout.isatty()
    except AttributeError:
        return False


@cache
def stderr_available() -> bool:
    try:
        return sys.stderr.isatty()
    except AttributeError:
        return False


# endregion


# region ---[ Data Serialization ]---
def loads_data(data: str, input_format: Format) -> dict:
    if input_format == "json":
        return json.loads(data)
    if input_format == "json5":
        return json5_loads(data)
    if input_format == "toml":
        return toml_loads(data)
    if input_format == "yaml":
        return yaml_loads(data)
    if input_format == "python":
        return eval(data)
    raise ValueError(
        f"Unsupported format: {input_format!r}. Please use one of {SUPPORTED_FORMATS_STR}"
    )


def dumps_data(data: dict, output_format: Format) -> str:
    if output_format == "json":
        return json_dumps(data)
    if output_format == "json5":
        return json5_dumps(data)
    if output_format == "toml":
        return toml_dumps(data)
    if output_format == "yaml":
        return yaml_dumps(data)
    if output_format == "python":
        return repr(data)
    raise ValueError(
        f"Unsupported format: {output_format!r}. Please use one of {SUPPORTED_FORMATS_STR}"
    )


def json_dumps(data) -> str:
    return json.dumps(data, indent=4, sort_keys=SETTINGS["sort_keys"])


def json5_loads(data: str) -> dict:
    return import_json5().loads(data)


def json5_dumps(data) -> str:
    return import_json5().dumps(data, indent=4, sort_keys=SETTINGS["sort_keys"])


def toml_loads(data: str) -> dict:
    return import_toml().loads(data)


def toml_dumps(data) -> str:
    # Does not support line width nor sorting keys.
    toml_module = import_toml()
    try:
        return toml_module.dumps(data)
    except TypeError as e:
        if not isinstance(data, list):
            raise
        if stdin_available() and stdout_available():
            answer = input(
                f"Got {e!r} while converting the data into toml. Data is a list, which is not supported by toml. "
                f'Make it a dict by converting to {{"data": data}}? [y/n] '
            )
            if answer.lower().strip() not in ("y", "yes"):
                raise
        return toml_module.dumps({"data": data})


def yaml_loads(data: str) -> dict:
    return import_yaml().load(data)


def yaml_dumps(data) -> str:
    yaml = import_yaml()
    import io

    output = io.StringIO()
    yaml.dump(data, output)
    return output.getvalue()


def python_collection_loads(data: str) -> dict:
    if not data.startswith("{") and not data.startswith("["):
        raise ValueError("Data does not start with '{' or '['")
    evaluated = eval(data)
    if not isinstance(evaluated, (dict, list)):
        raise TypeError(f"Data is not a dict or list: {evaluated!r}")
    return evaluated


# endregion


# region ---[ User Input Utils ]---
def load_input(
    input_arg: str, *, input_format: Optional[Format] = None
) -> tuple[str, Format]:
    """Returns a tuple of the input data and its format."""
    if input_arg == "-":
        assert is_piped(), "input arg is '-' but no data piped to stdin."
        data = read_stdin()
        format: Format = input_format if input_format else detect_format(data)
        return data, format
    if input_arg == "%paste":
        pyperclip = import_pyperclip()
        data = pyperclip.paste()
        format = input_format if input_format else detect_format(data)
        return data, format
    if not input_arg:
        if not is_piped():
            raise ValueError(
                "No input specified and stdin is not piped. Please provide input either as an argument or via stdin."
            )
        data = read_stdin()
        return data, input_format if input_format else detect_format(data)
    if input_path := is_file(input_arg):
        with open(input_path, "r") as f:
            data = f.read()
        return data, input_format if input_format else detect_format(data)
    data = input_arg
    return data, input_format if input_format else detect_format(data)


def detect_format(string: str) -> Format:
    try:
        json.loads(string)
        return "json"
    except json.JSONDecodeError:
        pass

    try:
        json5_loads(string)
        return "json5"
    except ValueError:
        pass
    except Exception as e:
        if "DecodeError" in repr(e):
            pass
        else:
            raise

    try:
        toml_loads(string)
        return "toml"
    except Exception as e:
        if "DecodeError" in repr(e):
            pass
        else:
            raise

    try:
        yaml_loads(string)
        return "yaml"
    except Exception as e:
        if "YAMLError" in repr(e):
            pass
        else:
            raise

    try:
        python_collection_loads(string)
        return "python"
    except Exception:
        pass

    raise ValueError(
        f"Input format {string!r} not recognized. Please use one of {SUPPORTED_FORMATS_STR}"
    )


def is_file(input_arg: Union[str, os.PathLike[str]]) -> Optional[Path]:
    try:
        path = Path(input_arg).expanduser()
        return path if path.is_file() else None
    except OSError:
        return None


# endregion

# ===[ "Business" Logic ]===
# ---[ Convert ]---

Serializable = typing.TypeVar(
    "Serializable",
    bound=Optional[Union[dict, list, str, int, bool]],
)


def convert(args: argparse.Namespace) -> str:
    input_arg = args.input
    input_format: Format = args.input_format
    output_format: Format = args.output_format
    output_dest = args.output
    clean: bool = args.clean
    pretty: bool = args.pretty
    if pretty and output_dest != STDOUT:
        raise ValueError("Pretty printing is only supported for standard output")
    SETTINGS["max_width"] = args.width
    SETTINGS["sort_keys"] = args.sort_keys
    stringified_data = _convert(
        input_arg,
        input_format=input_format,
        output_format=output_format,
        should_clean=clean,
        should_sort=SETTINGS["sort_keys"],
    )
    if output_dest == STDOUT:
        print_data_to_stdout(stringified_data, output_format, pretty=pretty)
    else:
        with open(output_dest, "w") as f:
            f.write(stringified_data)
        stderr_available() and stderr(f"✔ Data successfully written to {output_dest}")
    return stringified_data


def _convert(
    input_arg: str,
    *,
    input_format: Optional[Format] = None,
    output_format: Format,
    should_clean: bool,
    should_sort: bool,
) -> str:
    raw_input_data, input_format = load_input(input_arg, input_format=input_format)
    parsed_data: dict = loads_data(raw_input_data, input_format)
    if should_clean:
        parsed_data = clean_data(parsed_data)
    if should_sort:
        parsed_data = sort_data(parsed_data)
    stringified_data = dumps_data(parsed_data, output_format)
    return stringified_data


def clean_data(data: Serializable) -> Serializable:
    """Recursively remove keys with None or empty string values."""

    def _clean_dict(d: dict) -> dict:
        rv = {}
        for k, v in d.items():
            if v in (None, ""):
                continue
            _v = clean_data(v)
            if _v in (None, ""):
                continue
            rv[k] = _v
        return rv

    def _clean_sequence(s: list) -> list:
        return [_v for v in s if (_v := clean_data(v)) not in (None, "")]

    if isinstance(data, dict):
        return _clean_dict(data)
    if isinstance(data, list) and not isinstance(data, str):
        return _clean_sequence(data)

    return data


def sort_data(data: Serializable) -> Serializable:
    """Recursively sort dictionary keys or iterable values."""

    def _sort_dict(d: dict) -> dict:
        return {k: sort_data(v) for k, v in sorted(d.items())}

    def _sort_sequence(s: list) -> list:
        try:
            return sorted(sort_data(v) for v in s)
        except TypeError:
            return sorted((sort_data(v) for v in s), key=str)

    if isinstance(data, dict):
        return _sort_dict(data)
    if isinstance(data, list) and not isinstance(data, str):
        return _sort_sequence(data)

    return data


def print_data_to_stdout(formatted_data: str, output_format: Format, *, pretty: bool):
    if pretty:
        rich = import_rich()
        import rich.syntax

        console = rich.get_console()
        syntax = rich.syntax.Syntax(
            formatted_data,
            lexer=output_format,
            theme="monokai",
            line_numbers=True,
            background_color="rgb(0,0,0)",
            indent_guides=True,
            word_wrap=True,
        )
        console.print(syntax)
    else:
        print(formatted_data, flush=True)


# ---[ Diff ]---

DiffTool = Literal["diff", "delta", "code", "pycharm"]
SUPPORTED_DIFF_TOOLS: list[DiffTool] = DiffTool.__args__


def diff(args: argparse.Namespace) -> bool:
    """Return True if the two inputs are the same."""
    input1, input2 = args.input1, args.input2
    output_format: Format = args.output_format
    output_or_tool: Union[DiffTool, str] = args.output_or_tool
    input1_label = args.input1_label
    input2_label = args.input2_label
    quiet: bool = bool(args.quiet)
    ignore_order: bool = args.ignore_order
    ignore_empty: bool = args.ignore_empty
    ignore_space: bool = args.ignore_space
    if input1 == "-" and input2 == "-":
        raise argparse.ArgumentError(
            argument=None,
            message="Both inputs are '-', denoting stdin. Cannot diff two stdin inputs. At most one input can be stdin.",
        )
    if quiet and output_or_tool != "diff":
        raise argparse.ArgumentError(
            argument=None,
            message="Quiet mode is only supported with coreutils diff, e.g. -o diff.",
        )

    SETTINGS["sort_keys"] = ignore_order
    if ignore_space:
        SETTINGS["max_width"] = 9999
    stringified_data1: str = _convert(
        input1,
        output_format=output_format,
        should_clean=ignore_empty,
        should_sort=SETTINGS["sort_keys"],
    )
    stringified_data2: str = _convert(
        input2,
        output_format=output_format,
        should_clean=ignore_empty,
        should_sort=SETTINGS["sort_keys"],
    )

    if not input1_label:
        if input_path := is_file(input1):
            input1_label = input_path.name
        else:
            input1_label = "input1"

    if not input2_label:
        if input_path := is_file(input2):
            input2_label = input_path.name
        else:
            input2_label = "input2"

    input1_label, input2_label = input1_label + "_", input2_label + "_"

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f".{output_format}",
        prefix=input1_label,
        dir="/tmp",
        delete=False,
    ) as temp1, tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f".{output_format}",
        prefix=input2_label,
        dir="/tmp",
        delete=False,
    ) as temp2:
        temp1.write(stringified_data1)
        temp2.write(stringified_data2)
        temp1.file.flush()
        temp2.file.flush()
        temp1_path = str(Path(temp1.name))
        temp2_path = str(Path(temp2.name))

        write_to_file: bool = output_or_tool not in SUPPORTED_DIFF_TOOLS
        if write_to_file:
            output_path = output_or_tool
            return run_coreutils_diff(
                temp1_path,
                temp2_path,
                input1_label=input1_label,
                input2_label=input2_label,
                ignore_space=ignore_space,
                quiet=quiet,
                output_dest=output_path,
            )

        if output_or_tool == "diff":
            return run_coreutils_diff(
                temp1_path,
                temp2_path,
                input1_label=input1_label,
                input2_label=input2_label,
                ignore_space=ignore_space,
                quiet=quiet,
                output_dest=STDOUT,
            )

        if output_or_tool == "delta":
            subprocess_args = ["delta", temp1_path, temp2_path]
        elif output_or_tool == "code":
            subprocess_args = ["code", "--diff", temp1_path, temp2_path]
        elif output_or_tool == "pycharm":
            subprocess_args = ["pycharm", "diff", temp1_path, temp2_path]
        else:
            raise NotImplementedError(f"Diff tool {output_or_tool!r} not implemented.")

        joined_subprocess_args = shlex.join(subprocess_args)
        try:
            return os.system(joined_subprocess_args) == 0
        except Exception as e:
            stderr(f"❌ Error invoking '{' '.join(subprocess_args)}': {e!r}")
            return False


def run_coreutils_diff(
    temp1_path: str,
    temp2_path: str,
    *,
    input1_label: str,
    input2_label: str,
    ignore_space: bool,
    quiet: bool = False,
    output_dest: Optional[Union[str, Path]] = None,
) -> bool:
    """Return True if the inputs are the same."""
    whitespace_ignoring_options = [
        "-w",  # --ignore-all-blanks. For some reason the long option is unrecognized on macOS 14.5.
        "--strip-trailing-cr",
        "--ignore-blank-lines",
    ]
    coreutils_diff_args = [
        "diff",
    ]
    if quiet:
        coreutils_diff_args.append("-q")
    else:
        # --algorithm=patience or myers is not supported with --side-by-side, at least in macOS 14.5.
        coreutils_diff_args.append("--side-by-side")
    coreutils_diff_args.extend(
        [
            *(whitespace_ignoring_options if ignore_space else []),
            f"--label={input1_label}",
            f"--label={input2_label}",
            str(temp1_path),
            str(temp2_path),
        ]
    )
    if not output_dest or output_dest == STDOUT:
        coreutils_diff_args.append("--color=always")
        exitcode = os.system(shlex.join(coreutils_diff_args))
        if exitcode == 0:
            stderr("✔ Inputs data is identical.")
            return True
        return False
    else:
        coreutils_diff_args.append("--color=never")
        with open(output_dest, "w") as f:
            completed_process = subprocess.run(coreutils_diff_args, stdout=f)
            if completed_process.returncode == 0:
                stderr(f"✔ Inputs data is identical; Diff written to {output_dest}")
                return True
            stderr(f"✘ Inputs data is different; Diff written to {output_dest}")
            return False


# ===[ CLI ]===


def main():
    if sys.argv[1] not in ("convert", "diff"):
        sys.argv.insert(1, "convert")

    def argparse_formatter(prog):
        return argparse.RawTextHelpFormatter(prog=prog, max_help_position=40, width=80)

    parser = argparse.ArgumentParser(
        description="Convert or diff between JSON, YAML, TOML, JSON5 and literal Python collections.",
        formatter_class=argparse_formatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=False, metavar="command"
    )

    # Convert command
    convert_parser: argparse.ArgumentParser = subparsers.add_parser(
        "convert",
        help="Convert between JSON, YAML, TOML, JSON5 and literal Python collections.",
        formatter_class=argparse_formatter,
    )

    define_convert_arguments(convert_parser)

    # Diff command
    diff_parser: argparse.ArgumentParser = subparsers.add_parser(
        "diff",
        help="Diff between JSON, YAML, TOML, JSON5 and literal Python collections.",
        formatter_class=argparse_formatter,
    )

    define_diff_arguments(diff_parser)

    try:
        argcomplete.autocomplete(parser)
    except NameError:
        pass
    args: argparse.Namespace = parser.parse_args()
    if args.command is None:
        args.command = "convert"

    if args.command == "diff":
        no_difference: bool = diff(args)
        sys.exit(0 if no_difference else 1)
    elif args.command == "convert":
        return convert(args)
    else:
        parser.print_help()
        sys.exit(1)


def define_convert_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input",
        help="Data to convert. Can be a string, file path, '-' for stdin, or %%paste for clipboard content. If data is passed through stdin, this argument can be omitted",
        default="-",
        nargs="?",
    )
    parser.add_argument(
        "--input-format",
        help="Explicitly specify input format instead of auto-detection",
        required=False,
        choices=SUPPORTED_FORMATS,
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="output_format",
        required=True,
        choices=SUPPORTED_FORMATS,
        help="Format to convert the data into",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        required=False,
        default=STDOUT,  # Coupled to convert() not allowing pretty print with output file.
        help=f"Where to write the output. Either a file path or '{STDOUT}' for console output",
    )
    parser.add_argument(
        "--width",
        dest="width",
        required=False,
        type=int,
        default=SETTINGS["max_width"],
        help="Maximum line width. Currently only enforced in YAML",
    )
    parser.add_argument(
        "--sort-keys",
        default=SETTINGS["sort_keys"],
        action=argparse.BooleanOptionalAction,
        help="Sort dictionary key and list values in the output alphabetically",
    )
    parser.add_argument(
        "--clean",
        dest="clean",
        required=False,
        action="store_true",
        help="Remove keys and items with empty or null values from the output",
    )
    parser.add_argument(
        "-p",
        "--pretty",
        dest="pretty",
        required=False,
        action="store_true",
        help="Enable syntax highlighting and line numbers in console output",
    )


def define_diff_arguments(diff_parser: argparse.ArgumentParser) -> None:
    diff_parser.add_argument(
        "input1",
        help="First input to compare. Can be a string, file path, '-' for stdin, or %%paste for clipboard content. When using stdin for input1, input2 must not use stdin; similarily, when specifying %%paste for input1, input2 must not use %%paste.",
    )
    diff_parser.add_argument(
        "input2",
        help="Second input to compare. Can be a string, file path, '-' for stdin, or %%paste for clipboard content. When using stdin for input2, input1 must not use stdin; similarily, when specifying %%paste for input2, input1 must not use %%paste.",
    )
    diff_parser.add_argument(
        "-o",
        "--output",
        dest="output_or_tool",
        required=False,
        default="diff",
        help=(
            "Options:\n"
            f"- One of {SUPPORTED_FORMATS_STR}\n" + DIFF_OUTPUT_TOOLS_HELP_FORMATTED
        ),
    )
    diff_parser.add_argument(
        "-f",
        "--format",
        dest="output_format",
        required=True,
        choices=SUPPORTED_FORMATS,
        help="Show the diff in this format",
    )

    diff_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only return exit code (0=identical, 1=different) without showing diff output",
    )

    diff_parser.add_argument(
        "--input1-label",
        dest="input1_label",
        required=False,
        help="Custom label for the first input in diff output. Defaults to filename or 'input1_'",
    )

    diff_parser.add_argument(
        "--input2-label",
        dest="input2_label",
        required=False,
        help="Custom label for the second input in diff output. Defaults to filename or 'input2_'",
    )

    # These do not derive their default values from SETTINGS on purpose.
    #  Defaulting either to other than False is "insane defaults".
    diff_parser.add_argument(
        "--ignore-order",
        dest="ignore_order",
        required=False,
        action="store_true",
        default=False,
        help="Ignore the order of dict keys and list values when diffing",
    )
    diff_parser.add_argument(
        "--ignore-empty",
        dest="ignore_empty",
        required=False,
        action="store_true",
        default=False,
        help="Treat missing keys as equivalent to keys with empty/null values",
    )
    diff_parser.add_argument(
        "--ignore-space",
        dest="ignore_space",
        required=False,
        action="store_true",
        default=False,
        help="Ignore whitespace differences when comparing",
    )


if __name__ == "__main__":
    main()
