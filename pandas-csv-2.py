import pandas as pd
import datetime
from pandas import DataFrame

original_csv = pd.read_csv("F:\\test-data-ex\\text-out-5.csv")
#print(original_csv.values)
for i in original_csv.values:
    print(i)