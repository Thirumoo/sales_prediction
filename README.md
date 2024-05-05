import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('trrr.csv')

data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)

mode_of_out = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
miss_value = data['Outlet_Size'].isnull()
data.loc[miss_value, 'Outlet_Size'] = data.loc[miss_value, 'Outlet_Type'].apply(lambda x: mode_of_out[x])

data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)

encoder = LabelEncoder()
data['Item_Identifier'] = encoder.fit_transform(data['Item_Identifier'])
data['Item_Fat_Content'] = encoder.fit_transform(data['Item_Fat_Content'])
data['Item_Type'] = encoder.fit_transform(data['Item_Type'])
data['Outlet_Identifier'] = encoder.fit_transform(data['Outlet_Identifier'])
data['Outlet_Size'] = encoder.fit_transform(data['Outlet_Size'])
data['Outlet_Location_Type'] = encoder.fit_transform(data['Outlet_Location_Type'])
data['Outlet_Type'] = encoder.fit_transform(data['Outlet_Type'])

X = data.drop(columns='Item_Outlet_Sales')
y = data['Item_Outlet_Sales']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

train_prediction=regressor.predict(x_train)


from sklearn import metrics

r2_train=metrics.r2_score(y_train,train_prediction)

print('R_Squared values', r2_train)

