import torch
from torch import nn 
import torch.nn.functional as func
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from scipy.ndimage.interpolation import zoom

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 8) * (28 - 8), 10)
        )
    
    def forward(self, x):
        return self.model(x)

def RGBtoGray(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.144])

def main():
    st.title("MNIST Digit Predictor")
    l_col, r_col = st.columns(2)
    model = NN()
    model.load_state_dict(torch.load("./cnn3.pt"))


    with l_col:
        st.header("Draw a digit from 1-9")
        img = st_canvas(
            fill_color = "rgb(0, 0, 0)",
            stroke_width = 10,
            stroke_color = "#FFFFFF",
            background_color = "#000000",
            update_streamlit = True,
            width = 280,
            height = 280,
            drawing_mode = "freedraw",
            key = "canvas",
        )
    prediction = None
    if (img.image_data is not None):
        gray_img = RGBtoGray(img.image_data)
        gray_img = zoom(gray_img, 0.1) # 280*0.1 = 28
        x = torch.from_numpy(gray_img).unsqueeze(0)
        x = x.unsqueeze(0)
        x = x.float()

        output = model(x)
        
        prediction = torch.max(output, 1)
        prediction = prediction[1].numpy() # get the index
        
    with r_col: 
        st.header("Predicted:")
        # change below
        if prediction is not None: st.title(prediction[0])

if __name__ == "__main__":
    main()
