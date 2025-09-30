
import draccus
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

"""
Policy configuration classes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import draccus

@dataclass
class PolicyConfig(draccus.ChoiceRegistry):
    """Base configuration class for policies."""

@PolicyConfig.register_subclass("DiffusionPolicyConfig")
@dataclass
class DiffusionPolicyConfig(PolicyConfig):
    horizon: int = 16

@dataclass
class TrainingConfig:
    policy: PolicyConfig = field(default_factory=DiffusionPolicyConfig)
    

@draccus.wrap()
def main(cfg: TrainingConfig):
    print(cfg)


if __name__ == '__main__':
    main()