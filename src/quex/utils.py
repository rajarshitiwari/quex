"""
Utility functions
-----------------

"""

import copy
from collections import UserList
from typing import Generic, TypeVar

import numpy as np

# Generic Type variable
T = TypeVar("T")


class OpList(UserList, Generic[T]):
    """
    A list that triggers a callback whenever its contents are modified.
    Used to safely reset cached properties (like circuit layers) when
    the operations list is manipulated.
    """

    def __init__(self, callback, initial_data=None):
        super().__init__(initial_data)
        self._callback = callback

    # --- Mutating Methods Override ---
    def __setitem__(self, i, item):
        super().__setitem__(i, item)
        self._callback()

    def __delitem__(self, i):
        super().__delitem__(i)
        self._callback()

    def append(self, item):
        super().append(item)
        self._callback()

    def insert(self, i, item):
        super().insert(i, item)
        self._callback()

    def pop(self, i=-1):
        item = super().pop(i)
        self._callback()
        return item

    def extend(self, other):
        super().extend(other)
        self._callback()

    def remove(self, item):
        super().remove(item)
        self._callback()

    def clear(self):
        super().clear()
        self._callback()

    def __iadd__(self, other):
        """Catches the += operator"""
        super().__iadd__(other)
        self._callback()
        return self
