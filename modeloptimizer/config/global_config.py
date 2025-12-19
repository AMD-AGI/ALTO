from dataclasses import dataclass


@dataclass
class FeatureFlags:

    NON_LINEAR_INPUT_QUANTIZATION: bool = False
    CONCAT_TWO_INPUTS: bool = True
    AUTO_CAST_SIGNED_INT: bool = True


feature_flags = FeatureFlags()
