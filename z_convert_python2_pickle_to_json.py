import cPickle as pickle
import json

# Replace 'your_pickle_file.pkl' with the actual file path
pickle_file = r'D:\Coding\bodies-at-rest\lib_py\segmented_mesh_idx_faces.p'
json_file = r'json_file.json'

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

with open(json_file, 'w') as f:
    json.dump(data, f)