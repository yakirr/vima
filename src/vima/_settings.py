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

Plots that *are* drawn go through ``settings.show()``, which displays them when a
display is available and otherwise saves them as PNGs. This matters on a cluster:
with no display matplotlib falls back to the Agg backend, where ``plt.show()``
silently discards the figure and leaves it open, so the plots are lost and the
figures accumulate. Two knobs control the saving:

* ``settings.save_plots`` -- ``"auto"`` (default) saves only when no display is
  available, so notebooks are unaffected; ``True`` always saves (never displays),
  ``False`` never saves.
* ``settings.plot_dir`` -- directory for saved plots, created on demand; defaults
  to ``"figs"``. Set to ``None`` to discard plots instead (a warning is issued
  once).

Guidelines for package code:

* ``logger.info(...)``  -- normal progress messages (visible at ``default``).
* ``logger.debug(...)`` -- detailed diagnostics (visible only at ``verbose``).
* ``logger.warning(...)`` -- warnings the user should see at every level.
* ``settings.result(...)`` -- user-facing results shown at every level.
* ``settings.progress(iterable, name=...)`` -- wrap any loop needing a bar.
* ``if settings.show_plots(...):`` -- guard a diagnostic plot. Pass ``"verbose"``
  for the detailed plots; the default level guards the standard ones.
* ``settings.show(name)`` -- finish a figure. Never call ``plt.show()`` directly:
  that displays or discards, with no way to save.
"""

import os
import re
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

# matplotlib backends that render to a file and cannot display anything; a
# terminal with no display falls back to "agg". Names are matched lowercase.
# Fallback for the query below, and what matplotlib has reported for years.
_FILE_ONLY_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def _file_only_backends():
    """Names of the backends that can only write to a file.

    Asked of matplotlib where it exposes the list (3.9+), so a backend added
    later is picked up, with :data:`_FILE_ONLY_BACKENDS` as the fallback.

    Deliberately phrased as "which backends are file-only" rather than "which
    are interactive", even though matplotlib offers both: it classifies
    Jupyter's inline backend as *non*-interactive, since it drives no GUI event
    loop -- yet inline plots do appear in front of the user, which is the only
    thing we need to know. ``resolve_backend`` would answer that question, but
    it imports the backend module and raises on unknown names, so we avoid it.
    """
    try:
        from matplotlib.backends.registry import backend_registry, BackendFilter

        return set(backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE))
    except Exception:
        return _FILE_ONLY_BACKENDS

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
        self.plot_dir = "figs"
        self.save_plots = "auto"
        self._plot_count = 0
        self._last_figure = None
        self._warned_plots_discarded = False
        self._warned_save_failed = False
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

    def _display_available(self):
        """Whether the active matplotlib backend can actually show a figure.

        A backend is a display unless it is one of the file-only ones (see
        :func:`_file_only_backends`). Deciding it that way around means
        anything unrecognized -- a third-party or future backend, or a
        ``module://`` backend from an embedding host such as Jupyter's
        ``matplotlib_inline`` or ``ipympl`` -- counts as a display, so the
        worst case is plain ``plt.show()``, exactly as before this setting
        existed.
        """
        import matplotlib

        try:
            backend = matplotlib.get_backend()
        except Exception:  # backend resolution can fail in odd environments
            return True
        return backend.lower() not in _file_only_backends()

    def _saving_plots(self):
        """Whether :meth:`show` should save rather than display."""
        if self.save_plots == "auto":
            return not self._display_available()
        return bool(self.save_plots)

    def show(self, name=None, fig=None, overwrite=False):
        """Display the current figure, or save it when there is no display.

        Central replacement for ``plt.show()`` in package code. With a display
        available this is exactly ``plt.show()``. Without one -- a plain
        terminal on a cluster, where ``plt.show()`` silently discards the
        figure *and* leaves it open, so figures pile up -- the figure is
        written to ``settings.plot_dir`` as a PNG and closed.

        ``name`` labels the file, defaulting to the name of the calling
        function. Files are numbered in the order they are produced, so a run's
        plots sort chronologically; pass ``overwrite=True`` to use ``name``
        alone as the filename and replace the file on every call (for a plot
        redrawn repeatedly, like the per-epoch training summary). ``fig`` is the
        figure to save, defaulting to the current one -- pass it explicitly when
        drawing onto a figure that an earlier :meth:`show` may already have
        closed, since ``plt.gcf()`` would then hand back a blank one.

        Returns the path written, or ``None`` if the figure was displayed or
        discarded.
        """
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.gcf()
        self._last_figure = fig

        if not self._saving_plots():
            plt.show()
            return None

        if self.plot_dir is None:
            if not self._warned_plots_discarded:
                logger.warning(
                    "no display available and vima.settings.plot_dir is None, so "
                    "plots are being discarded; set vima.settings.plot_dir to save "
                    "them as images instead"
                )
                self._warned_plots_discarded = True
            plt.close(fig)
            return None

        if name is None:
            name = sys._getframe(1).f_code.co_name
        name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower() or "plot"
        if overwrite:
            filename = f"{name}.png"
        else:
            self._plot_count += 1
            filename = f"{self._plot_count:03d}_{name}.png"

        path = os.path.join(self.plot_dir, filename)
        try:
            os.makedirs(self.plot_dir, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
        except OSError as e:
            # A diagnostic plot is never worth aborting a long run for.
            if not self._warned_save_failed:
                logger.warning(f"could not save plot to {path}: {e}")
                self._warned_save_failed = True
            plt.close(fig)
            return None
        plt.close(fig)
        logger.info(f"saved plot to {path}")
        return path

    def current_figure(self):
        """The figure a follow-up call should draw on.

        Normally ``plt.gcf()``. When plots are being saved instead of shown,
        :meth:`show` closes each figure, so ``plt.gcf()`` would hand back a
        fresh blank one; with no figure open, return the last figure
        :meth:`show` handled instead.
        """
        import matplotlib.pyplot as plt

        if not plt.get_fignums() and self._last_figure is not None:
            return self._last_figure
        return plt.gcf()

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
