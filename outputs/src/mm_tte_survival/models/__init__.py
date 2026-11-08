"""Model layer. Legacy neural survival nets re-exported from .neural; the
first-class ResidualRiskModel and the abstract MultiHeadSurvivalModel live
alongside. The five OOP base classes (SurvivalHead / ModalityEncoder / FusionModule
plus DataSource and Modality in the data layer) and the composed ResistanceModel are
re-exported here so callers get one import surface."""
from .neural import SurvivalMLP, make_ema_teacher, update_ema_teacher
from .residual_risk import ResidualRiskModel
from .heads import SurvivalHead
from .encoders import ModalityEncoder
from .fusion import FusionModule, LateConcatFusion, CrossAttentionFusion, ResidualFusion
from .resistance_model import ResistanceModel

__all__ = [
    "SurvivalMLP", "make_ema_teacher", "update_ema_teacher", "ResidualRiskModel",
    "SurvivalHead", "ModalityEncoder", "FusionModule",
    "LateConcatFusion", "CrossAttentionFusion", "ResidualFusion", "ResistanceModel",
]
