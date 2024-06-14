import collections

try:
    from collections import abc
except ImportError:
    abc = collections  # Py 2.7

from mdtraj.core.topology import Atom, Residue

"""
Classes that use frozensets at keys, but allow access with by any iterable.

Contact maps frequently require mappings of pairs of objects (representing
the contact pair) to some value. Since the order of the objects in the pair
is unimportant (the pair (A,B) is the same as (B,A)), we use a ``set``.
However, since these are keys, the pair must be immutable: a ``frozenset``.
It gets really annoying to have to type ``frozenset`` around each object, so
the classes in this module allow other iterables (tuples, lists) to be used
as keys in getting/setting items -- internally, they are converted to
``frozenset``.
"""


class FrozenSetDict(abc.MutableMapping):
    """Dictionary-like object that uses frozensets internally.

    Note
    ----
    This can take input like ``dict({key: value})`` or
    ``dict([(key, value)])``, but not like ``dict(key=value)``, for the
    simple reason that in the last case, you can't use an iterable as key.
    """

    hash_map = frozenset
    def __init__(self, input_data=None):
        self.dct = {}
        if input_data is not None:
            if isinstance(input_data, collections.Mapping):
                # convert the mapping to key-value tuples
                input_data = list(input_data.items())

            for key, value in input_data:
                self.dct[self._regularize_key(key)] = value

    def __len__(self):
        return len(self.dct)

    def __iter__(self):
        return iter(self.dct)

    def _regularize_key(self, key):
        def all_isinstance(iterable, cls):
            return all(isinstance(k, cls) for k in iterable)

        if all_isinstance(key, Atom) or all_isinstance(key, Residue):
            key = self.hash_map(k.index for k in key)
        else:
            key = self.hash_map(key)

        return key

    def __getitem__(self, key):
        return self.dct[self._regularize_key(key)]

    def __setitem__(self, key, value):
        self.dct[self._regularize_key(key)] = value

    def __delitem__(self, key):
        del self.dct[self._regularize_key(key)]


def _make_frozen_set_counter(other):
    if not isinstance(other, FrozenSetCounter):
        other = FrozenSetCounter(other)
    return other


class FrozenSetCounter(FrozenSetDict):
    """Counter-like object that uses frozensets internally.
    """
    def __init__(self, input_data=None):
        if input_data is None:
            input_data = []

        if not isinstance(input_data, collections.Mapping):
            self.counter = collections.Counter([
                self._regularize_key(inp)
                for inp in input_data
            ])
        else:
            self.counter = collections.Counter({
                self._regularize_key(key): value
                for key, value in input_data.items()
            })

    def most_common(self, n=None):
        return self.counter.most_common(n)

    def elements(self):
        return self.counter.elements()

    def subtract(self, other):
        other = _make_frozen_set_counter(other)
        self.counter.subtract(other.counter)

    def update(self, other):
        other = _make_frozen_set_counter(other)
        self.counter.update(other.counter)

    def __add__(self, other):
        other = _make_frozen_set_counter(other)
        counter = self.counter + other.counter
        return FrozenSetCounter(counter)

    def __sub__(self, other):
        other = _make_frozen_set_counter(other)
        counter = self.counter - other.counter
        return FrozenSetCounter(counter)

    def __and__(self, other):
        pass

    def __or__(self, other):
        pass
