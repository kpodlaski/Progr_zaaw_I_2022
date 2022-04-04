import pandas as pd
import numpy
import numpy as np


data = pd.read_csv("../data/archiwum_tab_a_2021.csv",decimal=",", sep=";", encoding="cp1250")
print(data[['1EUR','1USD']][1:-3].values)
usd = data[['1USD']][1:-3].values.reshape(-1)
eur = data[['1EUR']][1:-3].values.reshape(-1)

print(numpy.corrcoef(usd,eur))

