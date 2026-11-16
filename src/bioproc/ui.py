"""User input for the notebook: ask, validate, record (thin UI layer).

Every decision the notebook needs from a person goes through :meth:`Session.ask`.
An answer is resolved in this order:

1. an answers file (``BIOPROC_ANSWERS=path/to/answers.json``), to replay a run;
2. the keyboard, when a person is running the notebook;
3. the default from the system config (``configs/systems/<name>.toml``, ``[answers]``).

When there is no keyboard (``nbclient``/``papermill`` execution, CI, pytest) the
``input()`` call raises, the session switches to non-interactive and the config default
is used. Each answer is recorded with where it came from, so the executed notebook
states which numbers a person chose and which were defaults.

Invalid input never passes silently: at the keyboard it is re-asked with the reason;
from a file or a default it raises ``ValueError`` (ground rules section 2).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_ATTEMPTS = 5
TRUE_WORDS = {"y", "yes", "true", "1", "on"}
FALSE_WORDS = {"n", "no", "false", "0", "off"}


def data_directory() -> Path:
    """Working directory for generated data and figures.

    ``BIOPROC_DATA_DIR`` if set, else ``~/bioproc-data``. Kept outside OneDrive and the
    repository (ground rules section 9).
    """
    path = Path(os.environ.get("BIOPROC_DATA_DIR", Path.home() / "bioproc-data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def short_path(path: str | Path) -> str:
    """Shorten a path for display: repo-relative, or ``~``-relative, with ``/``.

    The executed notebooks are committed, so anything they print is published. An
    absolute path from the machine that happened to run them tells a reader nothing
    about the analysis and quite a lot about that machine, so paths are shown relative
    to the repository or to the home directory instead.

    >>> short_path(Path.home() / "bioproc-data" / "demo")
    '~/bioproc-data/demo'
    """
    path = Path(path)
    repo = Path(__file__).resolve().parents[2]
    for base, prefix in ((repo, ""), (Path.home(), "~/")):
        try:
            return prefix + path.resolve().relative_to(base).as_posix()
        except ValueError:
            continue
    return path.as_posix()


@dataclass(frozen=True)
class Answer:
    key: str
    value: Any
    unit: str
    source: str  # "keyboard", "answers file", "config default", "default (no keyboard)"
    prompt: str


def parse_value(
    raw: Any,
    kind: type,
    *,
    choices: Sequence[str] | None = None,
    multiple: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Any:
    """Convert ``raw`` (text or an already-typed value) to ``kind`` and validate it.

    Choices accept the name or the 1-based number shown in the prompt. ``multiple``
    accepts a comma-separated list (or a list) and returns a list.

    >>> parse_value("2.5", float, minimum=0)
    2.5
    >>> parse_value("2", str, choices=["pls", "rf", "gp"])
    'rf'
    >>> parse_value("pls, gp", str, choices=["pls", "rf", "gp"], multiple=True)
    ['pls', 'gp']
    >>> parse_value("yes", bool)
    True

    Raises
    ------
    ValueError
        If the value cannot be converted, is not a listed choice, or is out of range.
    """
    if multiple:
        items = raw if isinstance(raw, list | tuple) else [p for p in str(raw).split(",") if p.strip()]
        if not items:
            raise ValueError("select at least one option")
        values = [parse_value(item, kind, choices=choices, minimum=minimum, maximum=maximum) for item in items]
        if len(set(map(str, values))) != len(values):
            raise ValueError(f"duplicate selections in {values}")
        return values

    if kind is bool:
        if isinstance(raw, bool):
            return raw
        word = str(raw).strip().lower()
        if word in TRUE_WORDS:
            return True
        if word in FALSE_WORDS:
            return False
        raise ValueError(f"expected yes/no, got {raw!r}")

    if choices is not None:
        text = str(raw).strip()
        if text.isdigit() and 1 <= int(text) <= len(choices) and text not in choices:
            return choices[int(text) - 1]
        if text not in choices:
            raise ValueError(f"{raw!r} is not one of {list(choices)}")
        return text

    if kind is str:
        text = str(raw).strip()
        if not text:
            raise ValueError("empty answer")
        return text

    try:
        number = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected a number, got {raw!r}") from error
    if kind is int and number != int(number):
        raise ValueError(f"expected a whole number, got {raw!r}")
    value = kind(number)
    if minimum is not None and value < minimum:
        raise ValueError(f"{value} is below the minimum {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{value} is above the maximum {maximum}")
    return value


class Session:
    """Asks the questions, remembers the answers.

    Parameters
    ----------
    defaults
        ``{key: default}``, normally the ``[answers]`` table of the system config.
    interactive
        ``True`` to prompt, ``False`` never to prompt, ``None`` (default) to prompt
        until the first time no keyboard is available. ``BIOPROC_INTERACTIVE=0/1``
        overrides it.
    answers_file
        JSON file of ``{key: value}`` replayed instead of prompting. ``BIOPROC_ANSWERS``
        overrides it.
    input_function
        Replaced in tests.
    """

    def __init__(
        self,
        defaults: dict[str, Any],
        *,
        interactive: bool | None = None,
        answers_file: str | Path | None = None,
        input_function: Callable[[str], str] = input,
    ) -> None:
        self.defaults = dict(defaults)
        env_interactive = os.environ.get("BIOPROC_INTERACTIVE")
        if env_interactive is not None:
            interactive = env_interactive.strip() == "1"
        self.interactive = interactive
        answers_file = os.environ.get("BIOPROC_ANSWERS", answers_file)
        self.replay: dict[str, Any] = {}
        if answers_file:
            with Path(answers_file).open(encoding="utf-8") as handle:
                self.replay = json.load(handle)
        self._input = input_function
        self.answers: list[Answer] = []
        self.notes: list[str] = []

    # ------------------------------------------------------------------------------
    def ask(
        self,
        key: str,
        prompt: str,
        kind: type = float,
        *,
        unit: str = "",
        choices: Sequence[str] | None = None,
        multiple: bool = False,
        minimum: float | None = None,
        maximum: float | None = None,
        help: str = "",  # noqa: A002 - reads naturally at the call site
        default: Any = None,
    ) -> Any:
        """Return a validated answer for ``key`` and record it.

        Raises
        ------
        KeyError
            If there is no default and no keyboard to ask.
        ValueError
            If a replayed answer or the default is invalid, or the keyboard gave
            ``MAX_ATTEMPTS`` invalid answers in a row.
        """
        options = {"choices": choices, "multiple": multiple, "minimum": minimum, "maximum": maximum}
        fallback = self.defaults.get(key, default)

        if key in self.replay:
            value = parse_value(self.replay[key], kind, **options)
            return self._record(key, value, unit, "answers file", prompt)

        if self.interactive is not False:
            label = self._label(prompt, unit, choices, multiple, fallback, help)
            for attempt in range(MAX_ATTEMPTS):
                try:
                    raw = self._input(label if attempt == 0 else "  try again: ")
                except (EOFError, NotImplementedError, RuntimeError, OSError):
                    # No stdin: nbclient raises StdinNotImplementedError (a
                    # NotImplementedError), pytest raises OSError, a closed pipe EOFError.
                    self.interactive = False
                    self.notes.append("No keyboard available; config defaults used from here on.")
                    break
                if raw is None or str(raw).strip() == "":
                    if fallback is None:
                        print("  an answer is required")
                        continue
                    value = parse_value(fallback, kind, **options)
                    return self._record(key, value, unit, "config default", prompt)
                try:
                    value = parse_value(raw, kind, **options)
                except ValueError as error:
                    print(f"  invalid: {error}")
                    continue
                return self._record(key, value, unit, "keyboard", prompt)
            else:
                raise ValueError(f"{key}: {MAX_ATTEMPTS} invalid answers in a row")

        if fallback is None:
            raise KeyError(f"{key}: no default in the config and no keyboard to ask")
        value = parse_value(fallback, kind, **options)
        return self._record(key, value, unit, "default (no keyboard)", prompt)

    def ask_path(self, key: str, prompt: str, *, suffixes: Sequence[str], allow_empty: bool = True) -> Path | None:
        """Ask for a file path; an empty answer returns ``None`` when allowed.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the suffix is not one of ``suffixes``.
        """
        raw = self.ask(key, prompt, str, default="none" if allow_empty else None, help=f"one of {list(suffixes)}")
        if allow_empty and raw.lower() in {"none", "-", ""}:
            return None
        path = Path(raw.strip().strip('"')).expanduser()
        if path.suffix.lower() not in suffixes:
            raise ValueError(f"{path.name}: expected one of {list(suffixes)}")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    # ------------------------------------------------------------------------------
    @staticmethod
    def _label(prompt, unit, choices, multiple, fallback, help_text) -> str:
        lines = [f"\n{prompt}" + (f" [{unit}]" if unit else "")]
        if help_text:
            lines.append(f"  {help_text}")
        if choices:
            lines.extend(f"  {i}. {c}" for i, c in enumerate(choices, start=1))
            if multiple:
                lines.append("  (several allowed, comma-separated)")
        shown = ", ".join(map(str, fallback)) if isinstance(fallback, list | tuple) else fallback
        lines.append(f"  Enter = {shown}" if fallback is not None else "  (required)")
        return "\n".join(lines) + "\n> "

    def _record(self, key: str, value: Any, unit: str, source: str, prompt: str) -> Any:
        self.answers = [a for a in self.answers if a.key != key]
        self.answers.append(Answer(key, value, unit, source, prompt))
        shown = ", ".join(map(str, value)) if isinstance(value, list) else value
        print(f"  {key} = {shown}{(' ' + unit) if unit else ''}   ({source})")
        return value

    def table(self):
        """Answers so far as a DataFrame."""
        import pandas as pd

        return pd.DataFrame([a.__dict__ for a in self.answers], columns=["key", "value", "unit", "source", "prompt"])

    def save(self, path: str | Path) -> Path:
        """Write ``{key: value}`` JSON that ``BIOPROC_ANSWERS`` can replay."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {a.key: a.value for a in self.answers}
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8", newline="")
        return path
