from SAInT.metrics.classification.hinge import hinge
from SAInT.metrics.classification.smoothed_hinge import smoothed_hinge
from SAInT.metrics.classification.squared_hinge import squared_hinge
from SAInT.metrics.classification.ramp import ramp
from SAInT.metrics.classification.modified_huber import modified_huber
from SAInT.metrics.classification.neglog_likelihood import neglog_likelihood
from SAInT.metrics.classification.cross_entropy import cross_entropy
from SAInT.metrics.classification.binary_cross_entropy import binary_cross_entropy

__all__ = ["hinge", "smoothed_hinge", "squared_hinge",
           "ramp", "modified_huber", "cross_entropy", "binary_cross_entropy",
           "neglog_likelihood"
           ]
