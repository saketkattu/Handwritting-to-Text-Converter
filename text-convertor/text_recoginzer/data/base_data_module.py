from pathlib import Path
from typing import Collecton,Dict,Optional,Tuple,Union
import argparse

from torch.utils.data import ConcatDataset,DataLoader
import pytorch_lighting as pl 

from text_recoginzer import utils
from text_recoginzer.data.utils import BaseDataset


