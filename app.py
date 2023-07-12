import torch
from torch import nn 
import torch.nn.functional as func
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
# from scipy.ndimage.interpolation import zoom

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 48, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(48, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10)
        )
