import functools

LAYER_MAPPINGS = {}
MODULE_QUANT_DEFAULTS = {}
QUANTIZERS = {}
OBSERVERS = {}


def register_layer_mapping(old_layer):

    @functools.wraps(old_layer)
    def decorator(new_layer):
        LAYER_MAPPINGS[old_layer] = new_layer

        assert old_layer.__name__ not in MODULE_QUANT_DEFAULTS, f"{old_layer.__name__} already exists in MODULE_QUANT_DEFAULTS"
        MODULE_QUANT_DEFAULTS[old_layer.__name__] = new_layer.default_params()

        return new_layer

    return decorator


def register_quantizers(cls):
    cls_name = cls.__name__
    if cls_name in QUANTIZERS:
        raise RuntimeError(f'Class {cls_name} has been registered already.')

    QUANTIZERS[cls_name] = cls
    return cls


def register_observers(cls):
    cls_name = cls.__name__
    if cls_name in OBSERVERS:
        raise RuntimeError(f'Class {cls} has been registered already.')
    OBSERVERS[cls_name] = cls
    return cls
