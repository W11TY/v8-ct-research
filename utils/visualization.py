import plotly.graph_objects as go
from skimage.measure import marching_cubes
import numpy as np

def generate_mesh(mask_data, spacing, color, name, opacity=1.0):
    """
    Generate a Plotly Mesh3d from a binary mask using marching cubes.
    Preserves physical aspect ratio by scaling vertices by voxel spacing.
    """
    if mask_data is None or np.sum(mask_data) == 0:
        return None
        
    # Marching cubes
    try:
        verts, faces, normals, values = marching_cubes(mask_data, level=0.5, spacing=spacing)
    except RuntimeError:
        return None
        
    # Plotly Mesh3d
    # Note: unpacking vertices into x, y, z arrays
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=opacity,
        color=color,
        name=name,
        showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.1, roughness=0.5)
    )
    
    return mesh

def create_3d_figure(meshes, title="3D View"):
    """
    Create a Plotly figure containing multiple 3D meshes.
    Ensures aspect ratio is maintained.
    """
    fig = go.Figure(data=[m for m in meshes if m is not None])
    
    # Update layout to preserve aspect ratio (Plotly usually defaults to a cube, 
    # we need 'data' aspectmode to keep the physical proportions correct)
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X (mm)', showbackground=False),
            yaxis=dict(title='Y (mm)', showbackground=False),
            zaxis=dict(title='Z (mm)', showbackground=False)
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

def get_2d_slice(volume_data, axis, index):
    """
    Get a 2D slice from the volume along the given axis.
    axis: 0 (Sagittal), 1 (Coronal), 2 (Axial) for typical NIfTI shapes, 
    but varies by orientation. We'll use 0, 1, 2 as indices.
    """
    if axis == 0:
        return volume_data[index, :, :]
    elif axis == 1:
        return volume_data[:, index, :]
    else:
        return volume_data[:, :, index]
