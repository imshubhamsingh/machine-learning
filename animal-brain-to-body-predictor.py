import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

datafram = pd.read_fwf('brain_body.txt')
x_value = datafram[['Brain']]
y_value = datafram[['Body']]

body_reg = linear_model.LinearRegression()
body_reg.fit(x_value, y_value)

plt.scatter(x_value, y_value)
plt.plot(x_value, body_reg.predict(x_value))
plt.show()
