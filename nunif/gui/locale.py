import sys
import locale
import warnings


def get_default_locale():
    # Need getdefaultlocale:
    # See https://github.com/python/cpython/issues/130796
    assert sys.version_info < (3, 15, 0)
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=DeprecationWarning, message="'locale.getdefaultlocale'*")
        return locale.getdefaultlocale()[0]
