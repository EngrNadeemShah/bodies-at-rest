import json
import pickle

# File paths
json_file = r'z_json_file.json'
pickle_file = r'z_converted_pickle_file.p'

# Load the data from the JSON file
with open(json_file, 'r') as f:
    data = json.load(f)

# Save the data to a pickle file
with open(pickle_file, 'wb') as f:
    pickle.dump(data, f)

print("Conversion successful.")