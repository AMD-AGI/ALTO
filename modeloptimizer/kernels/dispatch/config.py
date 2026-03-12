from dataclasses import dataclass


class TrainingOpBaseConfig:
    pass


@dataclass
class MXFP4TrainingOpConfig(TrainingOpBaseConfig):
    use_2dblock_x: bool
    use_2dblock_w: bool
    use_hadamard: bool
    use_sr_grad: bool
    use_dge: bool


@dataclass
class MXFP8TrainingOpConfig(MXFP4TrainingOpConfig):
    pass
