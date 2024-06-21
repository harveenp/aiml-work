import numpy as np
from google.cloud import storage

class FileReader:    
    def __init__(self,bucket_name):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
    
    def readNumpy(self,file_path):
        blob = self.bucket.blob(file_path)
        with blob_raw.open("rb") as f:
            data = np.load(f)
        return data
    
    def readText(self,file_path,skip_header=0,delimiter=',',filling_values=0):
        blob = self.bucket.blob(file_path)
        with blob.open("r") as f:
            data = np.genfromtxt(f,skip_header=skip_header,delimiter=delimiter,filling_values=filling_values)
        return data
