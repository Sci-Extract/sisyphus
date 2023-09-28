import pandas as pd

class X2C:
    def __init__(self,file_name):
        # just the name without suffix
        self.file_name = file_name

    def convert(self):
        file_read = self.file_name + ".xlsx"
        df = pd.read_excel(file_read)
        file_out = self.file_name + ".csv"
        df.to_csv(file_out, index=False) 
        