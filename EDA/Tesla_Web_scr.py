import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# input data
inputDF = pd.ExcelFile('Tesla_TrueCar_Used_Prices.xlsx')
tabnames = inputDF.sheet_names
inputdf = inputDF.parse(tabnames[0]).dropna()

models = inputdf.iloc[:,2:] 

#plt.figure(figsize=(8, 3))

inp = ['100D','P100D']

for i in inp:
    sns.kdeplot(models.ix[models['BatteryCapacity']==i, 'Mileage'], label='Mileage '+i, shade=True)

plt.title('Probability distribution of used Tesla model mileage')
plt.xlabel('Mileage')
plt.ylabel('Probability Density value')
plt.show()

for i in inp:
    sns.distplot(models.ix[models['BatteryCapacity']==i, 'Year'], label='Year '+i, kde=False)

plt.title('Probability distribution of used Tesla model dates')
plt.xlabel('Dates')
plt.ylabel('Probability Density value')
plt.show()

for i in inp:
    sns.kdeplot(models.ix[models['BatteryCapacity']==i, 'Price'], label='Price '+i, shade=True)

plt.title('Probability distribution of used Tesla prices')
plt.xlabel('Price')
plt.ylabel('Probability Density value')
plt.show()


#sns.distplot(models.ix[models['Year']==2017, 'BatteryCapacity'], label='Year '+i, kde=False)
models.ix[models['Year']==2016, 'BatteryCapacity'].hist()
plt.title('Proportion of cars on sales')
plt.xlabel('Model')
plt.ylabel('Counts')
plt.show()

# proportion of cars that turn up at used sales sites
car_set = pd.DataFrame(models.ix[models['Year']==2017, 'BatteryCapacity'])
battery_75 = car_set.ix[car_set['BatteryCapacity']=='75', 'BatteryCapacity'].count()/len(car_set.index)
battery_75D = car_set.ix[car_set['BatteryCapacity']=='75D', 'BatteryCapacity'].count()/len(car_set.index)
battery_100D = car_set.ix[car_set['BatteryCapacity']=='100D', 'BatteryCapacity'].count()/len(car_set.index)


