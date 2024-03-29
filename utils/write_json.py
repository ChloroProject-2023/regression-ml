import json
import os
import numpy as np
import re

def write_to_json(self, path, dimension, version, name_model):
    # convert all attributes to list
    for key, value in self.params.items():
        if isinstance(value, np.ndarray):
            self.params[key] = value.tolist()
        else:
            self.params[key] = value
    for key, value in self.met.items():
        if isinstance(value, np.ndarray):
            self.met[key] = value.tolist()
        else:
            self.met[key] = value
            
    for key, value in self.__dict__.items():
        if isinstance(value, np.ndarray):
            self.__dict__[key] = value.tolist()
        else:
            self.__dict__[key] = value
            
    filename = f'{name_model}_pca_{dimension}_{version}.json'
    filename = filename.replace(' ', '_')
    fullpath = os.path.join(path, filename)
    dict = {
        "name_model": self.name,
        "pca_dimension": dimension,
        "metrics": self.met,
        "params": self.params,
        "filepath": fullpath
    }
    
    if os.path.exists(fullpath):
        suffix = 1
        while os.path.exists(fullpath):
            fullpath = os.path.join(path, f'{name_model}_pca_{dimension}_{version}_{suffix}.json')
            suffix += 1
        dict['filepath'] = fullpath
        filename = re.search(r'[^/]+$', fullpath).group(0)
        dict['name_model'] = filename.replace('.json', '')
    
    fullpath = fullpath.replace(' ', '_')
    with open(fullpath, 'w') as f:
        json.dump(dict, f, indent=4)
    return fullpath.replace('\\', '/')