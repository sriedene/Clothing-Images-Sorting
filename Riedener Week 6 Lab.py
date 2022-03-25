from fbprophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)

#%%
#Note: RegionName column was changed to ZipCode prior to being read in for ease of comprehension
housing = pd.read_csv('C:\\Users\\User\\Desktop\\Syracuse\\Zip_Zhvi_SingleFamilyResidence.csv'
                      , encoding = 'latin-1')
#%%
print(housing.head())
housing.shape
housing = housing.dropna()
housing.shape
list(housing.columns)
metro_grouped = housing.groupby(['Metro', 'State'], as_index = False).mean()
print(metro_grouped.head(10))

# Selecting out desired AK data for plotting purposes
array = ['Hot Springs', 'Searcy', 'Little Rock-North Little Rock-Conway', 'Fayetteville-Springdale-Rogers']
ar_metro = metro_grouped.loc[(metro_grouped['Metro'].isin(array)) & (metro_grouped['State'] == 'AR')]
ar_metro = ar_metro.drop(columns=['RegionID','ZipCode','SizeRank','State'])
ar_metro_melt = pd.melt(ar_metro, id_vars = ['Metro'])
hotsprings = ar_metro_melt.loc[ar_metro_melt['Metro'] == 'Hot Springs']
searcy = ar_metro_melt.loc[ar_metro_melt['Metro'] == 'Searcy']
littlerock = ar_metro_melt.loc[ar_metro_melt['Metro'] == 'Little Rock-North Little Rock-Conway']
fayetteville = ar_metro_melt.loc[ar_metro_melt['Metro'] == 'Fayetteville-Springdale-Rogers']
#%%
#Code for creating line plots. Please run simultaneously and not one line at a time
plt.figure(figsize=(18,12))
plt.xticks(rotation=90)
plt.plot('variable', 'value', data = hotsprings)
plt.plot('variable', 'value', data = searcy)
plt.plot('variable', 'value', data = littlerock)
plt.plot('variable', 'value', data = fayetteville)

plt.legend(('Hot Springs', 'Searcy', 'Little Rock', 'Fayetteville'))
#%%
#Change the whole data set from wide format to long format
housing_melt = pd.melt(housing, id_vars = ['RegionID','ZipCode', 'City', 'State', 'Metro', 'CountyName', 'SizeRank'], var_name='Date', value_name='HousingValue')
housing_melt.shape

# Change date column to datetime data type
housing_melt['Date'] = pd.to_datetime(housing_melt['Date'])
housing_melt = housing_melt.sort_values(by=['ZipCode','Date'])

#%%
# Prediction code modified from example provided by student Jon Anderson
def prophetForecast(zipcode):
    df = housing_melt[housing_melt['ZipCode']==zipcode]
    df = df[['Date','HousingValue']]
    
    df.columns=['ds','y']
    
    my_model = Prophet(interval_width=.95)
    my_model.fit(df)
    future_dates = my_model.make_future_dataframe(periods=12, freq='M')
    
    forecast = my_model.predict(future_dates)
    forecast = forecast[['ds','yhat','yhat_lower','yhat_upper']]
    
    my_model.plot(forecast, uncertainty=True)
    return(forecast)
#%%
# Single out information for desired zip codes
# Save zip codes as a list for iterative purposes
rochester = housing_melt.loc[(housing_melt['Metro'] == 'Rochester') & (housing_melt['State'] == 'NY')]
rochester_zip = rochester.ZipCode.unique()
rochester_zip = rochester_zip.tolist()
syracuse = housing_melt.loc[(housing_melt['Metro'] == 'Syracuse') & (housing_melt['State'] == 'NY')]
syracuse_zip = syracuse.ZipCode.unique()
syracuse_zip = syracuse_zip.tolist()
buffalo = housing_melt.loc[(housing_melt['Metro'] == 'Buffalo-Cheektowaga-Niagara Falls') & (housing_melt['State'] == 'NY')]
buffalo_zip = buffalo.ZipCode.unique()
buffalo_zip = buffalo_zip.tolist()
#%%
# Run zip codes through for loop and store information in predictions to find highest value housing
predictions = []
for zipcode in rochester_zip:
    result = prophetForecast(zipcode)
    myindex=result.shape[0]
    predictions.append(result.iloc[myindex-1,1])
zip_predict = pd.DataFrame(
    {'ZipCode': rochester_zip,
     'HousingValue': predictions,
    })
zip_predict = zip_predict.sort_values(by=['HousingValue'])
print(zip_predict)
#%%
predictions2 = []
for zipcode in syracuse_zip:
    result = prophetForecast(zipcode)
    myindex=result.shape[0]
    predictions2.append(result.iloc[myindex-1,1])

zip_predict2 = pd.DataFrame(
    {'ZipCode': syracuse_zip,
     'HousingValue': predictions2,
    })
zip_predict2 = zip_predict2.sort_values(by=['HousingValue'])
print(zip_predict2)
#%%
predictions3 = []
for zipcode in buffalo_zip:
    result = prophetForecast(zipcode)
    myindex=result.shape[0]
    predictions3.append(result.iloc[myindex-1,1])
zip_predict3 = pd.DataFrame(
    {'ZipCode': buffalo_zip,
     'HousingValue': predictions3,
    })
zip_predict3 = zip_predict3.sort_values(by=['HousingValue'])
print(zip_predict3)
#%%
# This code would provide analysis for the entire data set
# Runtime would excede 8 hours
all_zip = housing_melt.ZipCode.unique()
all_zip = all_zip.tolist()
predictions4 = []

for zipcode in all_zip:
    result = prophetForecast(zipcode)
    myindex=result.shape[0]
    predictions4.append(result.iloc[myindex-1,1])

zip_predict4 = pd.DataFrame(
    {'ZipCode': all_zip,
     'HousingValue': predictions4,
    })
zip_predict4 = zip_predict4.sort_values(by=['HousingValue'])
print(zip_predict4)
#%%
# Used for checking on zip code at a time
zipcode = 14051
result = prophetForecast(zipcode)
print(result.tail(12))
myindex=result.shape[0]
prediction = result.iloc[myindex-1,1]
print(prediction)
#%%
# For clearing the plot space
plt.close(fig='all')
