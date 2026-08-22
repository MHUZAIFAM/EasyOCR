"""Shared test fixtures.

ttkbootstrap supports exactly one live application root per process and
raises if a second is created while the first is alive. Creating a fresh
root per test therefore fails intermittently, depending on whether the
previous one has finished tearing down. Its own guidance is to create one
root per process and reuse it, which is what the fixtures below do.
"""

import pytest

tk = pytest.importorskip("tkinter")
ttkb = pytest.importorskip("ttkbootstrap")


@pytest.fixture(scope="session")
def _tk_session_root():
    try:
        root = ttkb.Window(themename="tokyo-night-dark")
        root.withdraw()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def tk_root(_tk_session_root):
    """The shared root, emptied of any widgets the previous test left."""
    yield _tk_session_root
    for child in _tk_session_root.winfo_children():
        try:
            child.destroy()
        except tk.TclError:
            pass
    try:
        _tk_session_root.update()
    except tk.TclError:
        pass
