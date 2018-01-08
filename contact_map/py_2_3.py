import inspect

try:
    getargspec = inspect.getfullargspec
except AttributeError:
    getargspec = inspect.getargspec

def inspect_method_arguments(method, no_self=True):
    args = getargspec(method).args
    if no_self:
        args = [arg for arg in args if arg != 'self']
    return args

