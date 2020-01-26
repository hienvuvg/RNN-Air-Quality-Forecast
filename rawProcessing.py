from pandas import read_csv
import pandas as pd

df = read_csv('rawAVG_v2.csv')

# manually specify column names except index column
df.columns = ['node_id','DateTime','temperature','humidity','PM2_5data','PM10data','PM1data','AQI']

#counts = dataset['node_id'].value_counts() # Show the frequencies of node_id

# Sorting data by frequency of node_id
node_frequencies = pd.DataFrame(df.node_id.value_counts()) # create frequency dataframe
new_index = node_frequencies.merge(df[["node_id"]], left_index=True, right_on="node_id") # create similar dataframe to df with frequency
del new_index['node_id_x']  # delete redundant columns
del new_index['node_id_y']  # delete redundant columns

# output the original data frame in the order of the new index.
new_df = df.reindex(new_index.index)
df = new_df.reset_index(drop=True)

data = df.loc[df['node_id'] == 'ESP_00983468'] # choose data from a specific node

del data['node_id']
data.sort_values(by='DateTime', inplace=True) # Sort the data frame by date time
#print(data['DateTime'])

# Find the number of N/A and drop corresponding rows
numNA = data['AQI'].isna().sum()
print('No of N/A AQI: *******************************')
print(numNA) # print out
data = data[numNA:] # drop the rows

# Re-index the frame
datafm = data.reset_index(drop=True)

datafm.plot(subplots = True)

# save to file
datafm.to_csv('datasetAQI.csv')
datafmFinal = read_csv('datasetAQI.csv', index_col='DateTime')
datafmFinal.columns = ['index1','Temp','Humi','PM2_5','PM10','PM1','AQI']
del datafmFinal['PM10']
del datafmFinal['PM1']
del datafmFinal['AQI']
del datafmFinal['index1']

datafmFinal = datafmFinal.reset_index(drop=False)   # Reset index, add more column

df0 = datafmFinal['DateTime']
df1 = datafmFinal['Temp']
df2 = datafmFinal['Humi']
df3 = datafmFinal['PM2_5']

dfx = pd.DataFrame({'DateTime':df0,'PM2_5':df3,'Humi':df2,'Temp':df1})
dfx.sort_values(by='DateTime', inplace=True)
dfx = dfx.reset_index(drop=True)
dfx.set_index('DateTime', inplace=True)

dfx.plot(subplots = True)

dfx.Temp = dfx.Temp.diff()
dfx.Humi = dfx.Humi.diff()

dfx.plot(subplots = True)
dfx.to_csv('SampledPM2_5.csv')
