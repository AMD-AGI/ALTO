from dataclasses import dataclass
from functools import partial

@dataclass(unsafe_hash=True, kw_only=True, slots=True)
class TrainingOpBaseConfig:
    precision: str
    use_2dblock_x: bool
    use_2dblock_w: bool
    use_hadamard: bool
    use_sr_grad: bool
    use_dge: bool
    

MXFP4TrainingOpConfig = partial(TrainingOpBaseConfig, precision="mxfp4")

MXFP8TrainingOpConfig = partial(TrainingOpBaseConfig, precision="mxfp8")

SCHEMES = {
    "mxfp4": MXFP4TrainingOpConfig,
    "mxfp8": MXFP8TrainingOpConfig,
}

def get_scheme_config_class(scheme: str) -> type[TrainingOpBaseConfig]:
    if scheme not in SCHEMES:
        raise ValueError(f"Unsupported scheme: {scheme}")
    return SCHEMES[scheme]

def is_preset_scheme(scheme: str) -> bool:
    return scheme in SCHEMES
