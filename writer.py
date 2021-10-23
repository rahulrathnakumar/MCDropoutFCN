import numpy as np
import torch
import json



class Writer():
    '''
    This class writes experiment data to a file.
    '''
    def __init__(self, file_path):
        self.file_path = file_path
    def __write__(self, data):
        file = open(self.file_path)
        file.write
        