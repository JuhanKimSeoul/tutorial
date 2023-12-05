import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

print(os.getenv('KEY'))

s = pd.Series([1,2,3])

print(s)