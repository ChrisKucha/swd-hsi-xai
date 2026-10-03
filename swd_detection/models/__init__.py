from .cnn3d import CNN3D
from .cnn3d_transformer import CNN3DTransformer

MODEL_REGISTRY = {
    "cnn3d":             CNN3D,
    "cnn3d_transformer": CNN3DTransformer,
}

def build_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
