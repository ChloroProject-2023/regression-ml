import json
import os
import numpy as np

def write_to_json(self, path, dimension):
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
            
    filename = f'{self.name}_pca_{dimension}.json'
    filename = filename.replace(' ', '_')
    fullpath = os.path.join(path, filename)
    dict = {
        "name_model": self.name,
        "pca_dimension": dimension,
        "metrics": self.met,
        "params": self.params,
        "filepath": fullpath
    }
            
    with open(fullpath, 'w') as f:
        json.dump(dict, f, indent=4)
    return fullpath.replace('\\', '/')