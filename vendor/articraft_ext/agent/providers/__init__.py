"""Overlay onto the vendored Articraft ``agent.providers`` package.

Same trick as ``agent/__init__.py`` one level up: extending ``__path__`` hands
every module this directory does not define back to the official tree, so
``gpugeek.py`` can sit beside ``dashscope.py`` and import from
``agent.providers.chat_completions`` without forking anything.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
