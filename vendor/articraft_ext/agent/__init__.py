"""Overlay onto the vendored Articraft ``agent`` package.

``vendor/articraft_ext`` is placed ahead of ``vendor/articraft`` on ``sys.path``, so this
module is what Python finds first for the ``agent`` package. Extending ``__path__`` hands
the rest of the package back to the official tree, which lets the three modules here sit
next to (and import from) modules such as ``agent.compiler`` without forking them.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
