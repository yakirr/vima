"""Unified output control for vima.

All informational output in the package flows through a single ``logging``
logger (``vima._settings.logger``, named ``"vima"``) and a single progress-bar
helper (``settings.progress``). User-facing controls live on the module-level
``settings`` object, exposed as ``vima.settings``.

Three independent knobs:

* ``settings.verbosity`` -- how much informational output to emit. Three levels:

  ==================  ===  ================  ==================================
  name                int  logging level     what is shown
  ==================  ===  ================  ==================================
  ``"minimal"``       0    ``WARNING``       warnings/errors, progress bars,
                                             and results only (e.g. the
                                             ``association`` p-value)
  ``"default"``       1    ``INFO``          + high-level progress messages
  ``"verbose"``       2    ``DEBUG``         + detailed diagnostics
  ==================  ===  ================  ==================================

* ``settings.diagnostic_plots`` -- how many diagnostic plots to draw, on the same
  three-level scale as ``verbosity``: ``"minimal"`` draws none, ``"default"`` draws
  the plots historically shown by default (QC, mean-variance, embeddings), and
  ``"verbose"`` additionally draws the more detailed plots. Until it is set
  explicitly, it *tracks* ``verbosity`` -- so setting ``verbosity`` alone also
  controls the plots; assigning ``diagnostic_plots`` decouples the two.

* ``settings.progress_bars`` -- whether tqdm progress bars are shown. Orthogonal
  to verbosity (bars stay on even at ``"minimal"``); set ``False`` for batch or
  cluster runs.

Guidelines for package code:

* ``logger.info(...)``  -- normal progress messages (visible at ``default``).
* ``logger.debug(...)`` -- detailed diagnostics (visible only at ``verbose``).
* ``logger.warning(...)`` -- warnings the user should see at every level.
* ``settings.result(...)`` -- user-facing results shown at every level.
* ``settings.progress(iterable, name=...)`` -- wrap any loop needing a bar.
* ``if settings.show_plots(...):`` -- guard a diagnostic plot. Pass ``"verbose"``
  for the detailed plots; the default level guards the standard ones.
"""

import sys
import logging
from enum import IntEnum

from tqdm.auto import tqdm

__all__ = ["Verbosity", "settings", "logger"]


class Verbosity(IntEnum):
    """Amount of informational output emitted by vima."""

    minimal = 0  # warnings + progress bars + results only
    default = 1  # + high-level info messages (the historical default)
    verbose = 2  # + detailed diagnostics

    @classmethod
    def parse(cls, value):
        """Coerce an int, name string, or Verbosity into a Verbosity."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value]
            except KeyError:
                raise ValueError(
                    f"unknown verbosity {value!r}; "
                    f"expected one of {[v.name for v in cls]} or 0/1/2"
                )
        return cls(value)


_LOGGING_LEVELS = {
    Verbosity.minimal: logging.WARNING,
    Verbosity.default: logging.INFO,
    Verbosity.verbose: logging.DEBUG,
}

logger = logging.getLogger("vima")

# ANSI colors: debug messages are grayed out, results are green.
_GRAY, _GREEN, _RESET = "\033[90m", "\033[32m", "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Format log records, graying out DEBUG-level messages."""

    def format(self, record):
        msg = super().format(record)
        if record.levelno == logging.DEBUG:
            return f"{_GRAY}{msg}{_RESET}"
        return msg


class _TqdmLoggingHandler(logging.StreamHandler):
    """Route log records through ``tqdm.write`` so they never corrupt an active
    progress bar. When no bar is live, ``tqdm.write`` degrades to a plain write
    to the stream, so logging in loops without a bar is unaffected. This keeps
    the two output channels fully decoupled: progress bars honor
    ``settings.progress_bars`` and log messages honor ``settings.verbosity``,
    independent of each other.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream, end=self.terminator)
        except Exception:
            self.handleError(record)


class Settings:
    """Global vima output settings; access via the ``vima.settings`` singleton."""

    def __init__(self):
        # Attach a single stdout handler and keep our records off the root
        # logger so vima's output is independent of any host logging config.
        # The "vima" logger is a process-global singleton that survives module
        # reloads, so clear any handler we previously attached before adding a
        # new one; otherwise re-importing vima (e.g. importlib.reload in a
        # notebook) accumulates handlers and every message prints N times.
        # Match by class *name*, not isinstance: each reload defines a fresh
        # _TqdmLoggingHandler class, so handlers left by earlier reloads are
        # instances of a different (stale) class object and would fail an
        # isinstance check against the current class.
        for h in list(logger.handlers):
            if type(h).__name__ == "_TqdmLoggingHandler":
                logger.removeHandler(h)
        handler = _TqdmLoggingHandler(sys.stdout)
        handler.setFormatter(_ColorFormatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False

        self.progress_bars = True
        self._verbosity = None
        # ``diagnostic_plots`` tracks ``verbosity`` until the user sets it.
        self._diagnostic_plots = None
        self._diagnostic_plots_explicit = False
        self.verbosity = Verbosity.default

    @property
    def verbosity(self):
        """Current :class:`Verbosity` level (see module docstring)."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        self._verbosity = Verbosity.parse(value)
        logger.setLevel(_LOGGING_LEVELS[self._verbosity])
        if not self._diagnostic_plots_explicit:
            self._diagnostic_plots = self._verbosity

    @property
    def diagnostic_plots(self):
        """:class:`Verbosity` level controlling how many diagnostic plots are drawn.

        Tracks :attr:`verbosity` until assigned; assigning it decouples the two.
        """
        return self._diagnostic_plots

    @diagnostic_plots.setter
    def diagnostic_plots(self, value):
        self._diagnostic_plots = Verbosity.parse(value)
        self._diagnostic_plots_explicit = True

    def show_plots(self, level=Verbosity.default):
        """Whether diagnostic plots at ``level`` should be drawn.

        ``level`` is anything :meth:`Verbosity.parse` accepts (default the
        standard-plot level); pass ``"verbose"`` for the more detailed plots.
        """
        return self._diagnostic_plots >= Verbosity.parse(level)

    def progress(self, iterable=None, name=None, total=None, ncols=100, desc=None, **kwargs):
        """tqdm wrapper honoring ``settings.progress_bars``.

        Central replacement for the per-module ``pb = lambda ...`` helpers.
        ``name`` is a brief label for the loop, shown as the tqdm ``desc``
        prefix (``desc`` is still accepted as an alias; ``name`` wins).
        """
        using_widget = any(
            cls.__module__ == "tqdm.notebook"
            for cls in tqdm.mro()
        )
        return tqdm( 
            iterable, desc=name if name is not None else desc, total=total,
            ncols=ncols * (7 if using_widget else 1),
            bar_format='{l_bar}{bar}{r_bar}',
            disable=not self.progress_bars, **kwargs,
        )

    def result(self, message):
        """Emit a user-facing result, shown at every verbosity level."""
        tqdm.write(f"{_GREEN}{message}{_RESET}", file=sys.stdout)


settings = Settings()
