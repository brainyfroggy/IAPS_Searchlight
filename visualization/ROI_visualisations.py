import plotly.graph_objects as go
from nilearn import datasets, surface, image
from skimage import measure
from scipy import ndimage
import numpy as np
import os

# Your mask paths
amyg_mask = r"N:\Experimental_Data\Max Lobel\Pulivanar Deep Network Analysis\Amygdala Analysis\Masks\Amygdala_mask.nii"
pulv_mask = r"N:\Experimental_Data\Max Lobel\Pulivanar Deep Network Analysis\Pulvinar Analysis\Masks\rPulvinar.nii"

# Output directory
output_dir = r"N:\Experimental_Data\Max Lobel\Pulivanar Deep Network Analysis\SFN 2025"

# Define colors
amyg_color = 'red'
pulv_color = 'blue'
brain_color = 'lightgray'

# Color conversion function
def get_color_codes(color_name):
    color_map = {
        'red': {'hex': '#FF0000', 'rgb': (255, 0, 0)},
        'blue': {'hex': '#0000FF', 'rgb': (0, 0, 255)},
        'lightgray': {'hex': '#D3D3D3', 'rgb': (211, 211, 211)}
    }
    return color_map.get(color_name.lower(), {'hex': color_name, 'rgb': 'N/A'})

# Load masks
amyg_img = image.load_img(amyg_mask)
pulv_img = image.load_img(pulv_mask)

amyg_data = amyg_img.get_fdata()
pulv_data = pulv_img.get_fdata()

# Very light smoothing to reduce artifacts while preserving detail
amyg_data_smooth = ndimage.gaussian_filter(amyg_data, sigma=0.5)
pulv_data_smooth = ndimage.gaussian_filter(pulv_data, sigma=0.5)

# Create 3D meshes from volumetric data using marching cubes
def create_mesh_from_volume(volume_data, affine, threshold=0.5):
    """Convert volumetric mask to 3D mesh"""
    # Apply marching cubes to get surface
    verts, faces, normals, values = measure.marching_cubes(volume_data, level=threshold)
    
    # Transform vertices to MNI space
    verts_homogeneous = np.c_[verts, np.ones(len(verts))]
    verts_mni = verts_homogeneous.dot(affine.T)[:, :3]
    
    return verts_mni, faces

# Create meshes for ROIs with minimal smoothing and proper threshold
amyg_verts, amyg_faces = create_mesh_from_volume(amyg_data_smooth, amyg_img.affine, threshold=0.5)
pulv_verts, pulv_faces = create_mesh_from_volume(pulv_data_smooth, pulv_img.affine, threshold=0.5)

# Fetch fsaverage surface for brain
fsaverage = datasets.fetch_surf_fsaverage()

# Create figure
fig = go.Figure()

# Add both brain hemispheres (transparent)
for hemi in ['left', 'right']:
    coords, faces = surface.load_surf_mesh(fsaverage[f'pial_{hemi}'])
    
    fig.add_trace(go.Mesh3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=brain_color,
        opacity=0.1,
        name=f'Brain {hemi}',
        showlegend=False,
        hoverinfo='skip',
        lighting=dict(ambient=0.5, diffuse=0.8),
        flatshading=False
    ))

# Add 3D volumetric Amygdala mesh
fig.add_trace(go.Mesh3d(
    x=amyg_verts[:, 0],
    y=amyg_verts[:, 1],
    z=amyg_verts[:, 2],
    i=amyg_faces[:, 0],
    j=amyg_faces[:, 1],
    k=amyg_faces[:, 2],
    color=amyg_color,
    opacity=1.0,
    name='Amygdala',
    showlegend=True,
    hoverinfo='name',
    lighting=dict(ambient=0.7, diffuse=1.0, specular=0.3, roughness=0.5),
    flatshading=False
))

# Add 3D volumetric Pulvinar mesh
fig.add_trace(go.Mesh3d(
    x=pulv_verts[:, 0],
    y=pulv_verts[:, 1],
    z=pulv_verts[:, 2],
    i=pulv_faces[:, 0],
    j=pulv_faces[:, 1],
    k=pulv_faces[:, 2],
    color=pulv_color,
    opacity=1.0,
    name='Pulvinar',
    showlegend=True,
    hoverinfo='name',
    lighting=dict(ambient=0.7, diffuse=1.0, specular=0.3, roughness=0.5),
    flatshading=False
))

# Layout
fig.update_layout(
    title='Amygdala (Red) & Pulvinar (Blue) - Full 3D Volumes',
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor='white',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        ),
        aspectmode='data'
    ),
    paper_bgcolor='white',
    width=1400,
    height=1000
)

# Save
fig.write_html(os.path.join(output_dir, 'plotly_3D_volumetric_ROIs_no_labels.html'))

# Print color codes for PowerPoint
print(f"\n{'='*60}")
print(f"COLOR CODES FOR POWERPOINT")
print(f"{'='*60}\n")

amyg_codes = get_color_codes(amyg_color)
print(f"AMYGDALA:")
print(f"  Hex Code:  {amyg_codes['hex']}")
print(f"  RGB:       {amyg_codes['rgb']}")
print()

pulv_codes = get_color_codes(pulv_color)
print(f"PULVINAR:")
print(f"  Hex Code:  {pulv_codes['hex']}")
print(f"  RGB:       {pulv_codes['rgb']}")
print()

brain_codes = get_color_codes(brain_color)
print(f"BRAIN SURFACE:")
print(f"  Hex Code:  {brain_codes['hex']}")
print(f"  RGB:       {brain_codes['rgb']}")
print()

print(f"{'='*60}")
print(f"Visualization saved to: plotly_3D_volumetric_ROIs_no_labels.html")
print(f"No labels - ready for PowerPoint annotation!")
print(f"{'='*60}\n")