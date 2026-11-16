# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from torchtitan.models.flux.trainer import FluxTrainer as TitanFluxTrainer

from alto.train import Trainer as AltoTrainer


class FluxTrainer(TitanFluxTrainer, AltoTrainer):
    """Flux encoder setup with ALTO's training step.

    ``TitanFluxTrainer`` comes first so its ``__init__`` loads the autoencoder,
    CLIP, and T5 after the base trainer exists. It does not define
    ``train_step``, so that method is ALTO's: training mode delegates to
    TorchTitan, and a forward-only recipe still runs calibration.
    """
