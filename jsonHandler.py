import numpy as np
import json
import os    
import tempfile
import itertools as IT


######### from https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
def uniquify(path, sep = ''):
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s = sep, n = next(count))
    orig = tempfile._name_sequence 
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir = dirname, prefix = filename, suffix = ext)
        tempfile._name_sequence = orig
    return filename
###################################################################################################


def save_json(data, filename):
    # convert ndarray to list
    for field in data:
        if isinstance(data[field], np.ndarray):
            data[field] = data[field].tolist()
    
    filename = uniquify(filename)
    try:
        with open(filename, 'w') as out_file:
            json.dump(data, out_file)        
    except Exception as e:
        print('Could not create new file')
        print(str(e))