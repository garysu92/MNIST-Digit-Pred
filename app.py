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
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 8) * (28 - 8), 10)
        )
    
    def forward(self, x):
        return self.model(x)

def main():
    st.title("MNIST Digit Predictor")
    l_col, r_col = st.columns(2)

    with l_col:
        st.header("Draw a digit from 1-9")
        img = st_canvas(
            fill_color="rgb(0, 0, 0)",
            stroke_width = 10,
            stroke_color="#FFFFFF",
            background_color="#000000",
            update_streamlit=True,
            width=224,
            height=224,
            drawing_mode="freedraw",
            key="canvas",
        )

    with r_col: 
        st.header("Predicted:")
        # change below
        st.title("Something...")

if __name__ == "__main__":
    main()
