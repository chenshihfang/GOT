# The core code logic was originally implemented by a human developer.
# Codex was used for post-publication refactoring, cleanup, and code quality improvements.
"""GOT-Edit tracker entry point, preserving the public ToMP tracker class."""

from .tomp import ToMP


def get_tracker_class():
    return ToMP
