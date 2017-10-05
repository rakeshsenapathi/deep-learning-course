import pandas as pd 
from sklearn import linear_model
import matplotlib.pyplot as mlib

#read the data
df = pd.read_fwf("brain_body.txt")
x_values = df[['Brain']]
y_values = df[['Body']] 


#Training model on the data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values) #gives a regression line that best fits

#Plotting
mlib.scatter(x_values, y_values)
mlib.plot(x_values, body_reg.predict(x_values))
mlib.show()