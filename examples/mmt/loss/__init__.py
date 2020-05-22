from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy
from .oim import OIMLoss

__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy',
    'OIMLoss'
]
