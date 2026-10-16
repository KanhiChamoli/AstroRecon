# AstroRecon: 3D Volumetric Nebular Reconstruction

## Mathematical Formulation

### 1. Forward Ray Marcher Model (Optically Thin Radiative Transfer)
$$I_{\text{synthetic}}(u, v) = \sum_{t} V(x(t), y(t), z(t)) \cdot \Delta t$$

### 2. Multi-Term Loss Function (Data Fit + Physical Regularization)
$$\mathcal{L}_{\text{total}} = \|\hat{I} - I_{\text{observed}}\|^2_2 + \lambda_{\text{TV}} \mathcal{L}_{\text{TV}}(V) + \lambda_{\text{MHD}} \|\nabla \cdot \mathbf{B}\|^2$$