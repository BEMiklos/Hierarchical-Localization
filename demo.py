import tqdm, tqdm.notebook
tqdm.tqdm = tqdm.notebook.tqdm
from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive
from hloc.utils import viz_3d
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Define input/output paths
images = Path('datasets/sacre_coeur')
outputs = Path('outputs/demo/')
sfm_pairs = outputs / 'pairs-sfm.txt'
loc_pairs = outputs / 'pairs-loc.txt'
sfm_dir = outputs / 'sfm'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'

# Configure feature and matcher
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

# Define the images to use for mapping
references = [p.relative_to(images).as_posix() for p in (images / 'mapping/').iterdir()]

# Extract features and match them across image pairs
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,profile_memory=True,use_cuda=True) as prof:
    with record_function("extract_features_main"):
        extract_features.main(feature_conf, images, image_list=references, feature_path=features)
    with record_function("pairs_from_exhaustive_main"):
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    with record_function("match_features_main"):
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
prof.export_chrome_trace(str(outputs / 'prof/extract_features.json'))  
print(prof.key_averages().table(sort_by="cuda_time_total"))        

# Run incremental Structure-From-Motion and display the reconstructed 3D model
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,profile_memory=True,use_cuda=True) as prof:
    with record_function("reconstruction_main"):
        model = reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references)
prof.export_chrome_trace(str(outputs / 'prof/reconstruction_calc.json'))  
print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))  

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping")
prof.export_chrome_trace(str(outputs / 'prof/reconstruction.json'))  
print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))  

# Define the query image for localization
query = 'query/night.jpg'
url = "https://upload.wikimedia.org/wikipedia/commons/5/53/Paris_-_Basilique_du_Sacr%C3%A9_Coeur%2C_Montmartre_-_panoramio.jpg"
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,profile_memory=True,use_cuda=True) as prof:
    extract_features.main(feature_conf, images, image_list=[query], feature_path=features, overwrite=True)
    pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references)
    match_features.main(matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True)
prof.export_chrome_trace(str(outputs / 'prof/query_extract.json'))  
print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))  

# Estimate the camera pose of the query image using PnP+RANSAC and refine the camera parameters
import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

camera = pycolmap.infer_camera_from_image(images / query)
ref_ids = [model.find_image_with_name(r).image_id for r in references]
conf = {
    'estimation': {'ransac': {'max_error': 12}},
    'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
}
localizer = QueryLocalizer(model, conf)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,profile_memory=True,use_cuda=True) as prof:
    with record_function("pose_from_cluster"):
        ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)
prof.export_chrome_trace(str(outputs / 'prof/pose_estimation.json'))  
print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))       

# Print the number of inlier correspondences found and visualize the correspondences
print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
    

