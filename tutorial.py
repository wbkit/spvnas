# %% [markdown]
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "spvnas"))

import numpy as np

import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate

import plotly.graph_objs as go
import plotly.io as pio

from model_zoo import spvnas_specialized


COLOR_MAP = np.array(
    [
        "#f59664",
        "#f5e664",
        "#963c1e",
        "#b41e50",
        "#ff0000",
        "#1e1eff",
        "#c828ff",
        "#5a1e96",
        "#ff00ff",
        "#ff96ff",
        "#4b004b",
        "#4b00af",
        "#00c8ff",
        "#3278ff",
        "#00af00",
        "#003c87",
        "#50f096",
        "#96f0ff",
        "#0000ff",
        "#ffffff",
    ]
)

LABEL_MAP = np.array(
    [
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        0,
        1,
        19,
        19,
        19,
        2,
        19,
        19,
        3,
        19,
        4,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        5,
        6,
        7,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        8,
        19,
        19,
        19,
        9,
        19,
        19,
        19,
        10,
        11,
        12,
        13,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        14,
        15,
        16,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        17,
        18,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
        19,
    ]
)

# %% [markdown]
# Let's load some real lidar data and pre-process it for inference.

# %%
# Load the LiDAR data and its label
lidar = np.fromfile("./assets/000000.bin", dtype=np.float32)
label = np.fromfile("./assets/000000.label", dtype=np.int32)
lidar = lidar.reshape(-1, 4)
label = LABEL_MAP[label & 0xFFFF]

# Filter out ignored points
lidar = lidar[label != 19]
label = label[label != 19]

# Quantize coordinates
coords = np.round(lidar[:, :3] / 0.05)
coords -= coords.min(0, keepdims=1)
feats = lidar

# Filter out duplicate points
coords, indices, inverse = sparse_quantize(coords, return_index=True, return_inverse=True)
coords = torch.tensor(coords, dtype=torch.int)
feats = torch.tensor(feats[indices], dtype=torch.float)

inputs = SparseTensor(coords=coords, feats=feats)
inputs = sparse_collate([inputs]).cuda()

# Load the pre-trained model from model zoo
model = spvnas_specialized("SemanticKITTI_val_SPVNAS@65GMACs").cuda()
model.eval()

# Run the inference
outputs = model(inputs)
outputs = outputs.argmax(1).cpu().numpy()

# Map the prediction back to original points
outputs = outputs[inverse]


# %% [markdown]
# Finally, we visualize the predictions from SPVNAS in as an HTML an interactive window. Enjoy!
pio.renderers.default = "browser"

trace = go.Scatter3d(
    x=lidar[:, 0],
    y=lidar[:, 1],
    z=lidar[:, 2],
    mode="markers",
    marker={
        "size": 2,
        "opacity": 1,
        "color": COLOR_MAP[outputs].tolist(),
    },
)

# Fix the aspect ratio in plotly
HEIGHT_SCALE = 3
x_aspect = 1
y_aspect = np.max(lidar[:, 1]) / np.max(lidar[:, 0])
z_aspect = np.max(lidar[:, 2]) / np.max(lidar[:, 0]) * HEIGHT_SCALE

layout = go.Layout(
    margin={"l": 0, "r": 0, "b": 0, "t": 0},
    scene=dict(
        aspectmode="manual",
        aspectratio=dict(x=x_aspect, y=y_aspect, z=z_aspect),
        xaxis=dict(backgroundcolor="black"),
        yaxis=dict(backgroundcolor="black"),
        zaxis=dict(backgroundcolor="black", visible=False),
        bgcolor="black",
        camera=dict(eye=dict(x=1.1, y=1.1, z=1.1)),
    ),
)

go.Figure(data=[trace], layout=layout).write_html("file.html")
  