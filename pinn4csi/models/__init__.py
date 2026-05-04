# pyright: basic, reportMissingImports=false, reportUnknownVariableType=false

"""Neural network model architectures."""

from pinn4csi.models.backprojection import ClassicalBackprojection
from pinn4csi.models.csi_pinn import CSIPhysicsAutoencoder, LossToggles
from pinn4csi.models.domain_invariant import (
    DomainInvariantLossToggles,
    PhysicsDomainInvariantModule,
    coral_loss,
    residual_moment_alignment_loss,
)
from pinn4csi.models.neural_operator import (
    PhysicsInformedDeepONet,
    create_physics_informed_deeponet,
)
from pinn4csi.models.paper1_models import (
    CSIAutoencoderClassifier,
    CSIClassifierMLP,
    CSICNNClassifier,
    CSICNNGRUClassifier,
    CSIDGSenseLiteClassifier,
    Paper1Model,
    Paper1ModelFactoryConfig,
    Paper1ModelSpec,
    create_paper1_model,
    expand_paper1_model_specs,
    list_paper1_model_names,
)
from pinn4csi.models.paper2_baselines import (
    DomainAdaptationBaselineConfig,
    Paper2DomainAdaptationBaseline,
    create_domain_adaptation_baseline,
    list_domain_adaptation_baselines,
)
from pinn4csi.models.pinn import PINN
from pinn4csi.models.wifi_imager import WiFiImagingPINN

__all__ = [
    "PINN",
    "WiFiImagingPINN",
    "ClassicalBackprojection",
    "CSIPhysicsAutoencoder",
    "LossToggles",
    "PhysicsDomainInvariantModule",
    "PhysicsInformedDeepONet",
    "DomainInvariantLossToggles",
    "coral_loss",
    "residual_moment_alignment_loss",
    "create_physics_informed_deeponet",
    "Paper2DomainAdaptationBaseline",
    "DomainAdaptationBaselineConfig",
    "create_domain_adaptation_baseline",
    "list_domain_adaptation_baselines",
    "Paper1Model",
    "Paper1ModelFactoryConfig",
    "Paper1ModelSpec",
    "CSIClassifierMLP",
    "CSICNNClassifier",
    "CSICNNGRUClassifier",
    "CSIDGSenseLiteClassifier",
    "CSIAutoencoderClassifier",
    "create_paper1_model",
    "expand_paper1_model_specs",
    "list_paper1_model_names",
]
