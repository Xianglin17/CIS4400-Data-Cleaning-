#!/usr/bin/env python
# coding: utf-8

# In[1]:


####################################################### Get Data from API ###################################################################################
pip install sodapy


# In[2]:


pip install pandas


# In[3]:


import pandas as pd
from sodapy import Socrata
import requests
import json


# In[4]:


#Key ID bbj70r9rfwf90nfzhyaugbpz0

#Key Secrete 4yau68jwil77nqk9cc2jhngt49j1gtujobvn873yzenhvisn73


# In[9]:


data_url='data.cityofnewyork.us'     # The Host Name for the API endpoint (the https:// part will be added automatically)
data_set='erm2-nwe9'    # The data set at the API endpoint (311 data in this case)
app_token='VZnU8GsLKScwiVCCxr4F7Z6DH'   # The app token created in the prior steps
client = Socrata(data_url,app_token)      # Create the client to point to the API endpoint
# Set the timeout to 60 seconds    
client.timeout = 60


# In[24]:


# Retrieve the results returned as JSON object from the API
# The SoDaPy library converts this JSON object to a Python list of dictionaries
results1 = client.get(data_set, where="complaint_type = 'Damaged Tree'")
results1 = pd.DataFrame.from_records(results1)
results2 = client.get(data_set, where="complaint_type = 'Dead/Dying Tree'")
results2 = pd.DataFrame.from_records(results2)
results3 = client.get(data_set, where="complaint_type = 'Illegal Tree Damage'")
results3 = pd.DataFrame.from_records(results3)


# In[27]:


combined_df = pd.concat([results1,results2],ignore_index=True)
final_df = pd.concat([combined_df,results3],ignore_index=True)


# In[42]:


TreeRecovery = final_df[['unique_key','created_date','closed_date','agency','agency_name','complaint_type','status','city','borough','incident_zip','latitude','longitude']]
TreeRecovery.head()


# In[43]:


# Save the data frame to a CSV file
TreeRecovery.to_csv("TreeRecovery.csv")


# In[50]:


pip install py2nb


# In[ ]:


################################################ K means clustering for weather #################################################################################


# In[616]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

get_ipython().run_line_magic('matplotlib', 'inline')


# In[733]:


features = ['Temperature_M','Temperature_Avg', 'Temperature_Max', 'Wdspeed_M','Wdspeed_Avg','Wdspeed_Max','Precipitation_Total']
data = pd.read_csv('weather_data_5_boroughs_daily.csv')
for feature in features:
    data[feature] = data[feature].str.strip(' ').astype(float)


# In[734]:


data.insert(loc=0, column='rowID', value=np.arange(len(data)))


# In[735]:


sampled_df = data[(data['rowID'] % 10) == 0]
sampled_df.shape


# In[736]:


sampled_df[features]


# In[737]:


select_df = sampled_df[features]
Test = data[features]


# In[1105]:


Test.describe()


# In[739]:


select_df.columns


# In[740]:


X = select_df


# In[741]:


kmeans = KMeans(n_clusters = 7)
model = kmeans.fit(X)
print("model\n", model)


# In[742]:


centers = model.cluster_centers_
centers


# In[743]:


def pd_centers(featuresUsed, centers):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	# Zip with a column called 'prediction' (index)
	Z = [np.append(A, index) for index, A in enumerate(centers)]

	# Convert to pandas data frame for plotting
	P = pd.DataFrame(Z, columns=colNames)
	P['prediction'] = P['prediction'].astype(int)
	return P
P = pd_centers(features, centers)
P


# In[628]:


predict = model.predict(Test)
select_df.columns


# In[1228]:


Test = data[features]
Test['Level'] = predict


# In[1229]:


Test.insert(loc=0, column='rowID', value=np.arange(len(Test)))


# In[1230]:


Test.describe()


# In[1231]:


Test.loc[0,features]


# In[1232]:


Type = []
for row in range(len(Test)):
    if Test['Temperature_Avg'][row] <= 48.529245 or Test['Temperature_M'][row] <= 41.511321 and Test['Temperature_Max'][row] <= 55.975472:
        if Test['Precipitation_Total'][row] <= 0.219811 and Test['Precipitation_Total'][row] > 0 :
            Type.append('Light Snow')
            
            
        elif Test['Precipitation_Total'][row] >  0.219811:
                if Test['Wdspeed_Max'][row] > 28.218868:
                    Type.append('SnowStorm')
                else:
                    Type.append('Heavy_Snow')
            
        else :
            if Test['Wdspeed_Avg'][row] >= 5.300000 and Test['Wdspeed_Max'][row]>= 19.900000 :
                Type.append('Windy')
            else:
                Type.append('Sunny')

    else:
        if Test['Precipitation_Total'][row] < 0.219811 and Test['Precipitation_Total'][row] > 0:
            Type.append('Light_Rain')
            
        elif Test['Precipitation_Total'][row] > 0.219811:
            if Test['Wdspeed_Max'][row] > 28.218868:              
                Type.append('Storm') 
            else:
                Type.append('Heavy_Rain')
            
        else:
            if Test['Wdspeed_Avg'][row] >= 5.300000 and Test['Wdspeed_Max'][row]>= 19.900000 :
                Type.append('Windy')
                
            else:
                Type.append('Sunny')
            
Test['Weather Type'] = Type


# In[1233]:


data.rename(columns={'rowID':'WeatherID'}, inplace = True)
Test.rename(columns={'rowID':'WeatherID'}, inplace = True)


# In[1234]:


Final = pd.merge(data,Test[['WeatherID','Level','Weather Type']],how='inner',on=['WeatherID'])
Final


# In[2]:


Replace = ['Level 2','Level 1','Level 4','Level 1','Level 2','Level 1','Level 3']

for number in range(7):   
    Final['Level'].replace(number, Replace[number], inplace=True)

Final


# In[1236]:


Final.to_csv("TreeRecovery_WeatherFinal.csv")


# In[1244]:


f = 0
for row in range(len(Final)):
    if Final['Weather Type'][row] == "Storm":
         f+=1
f


# In[10]:


################################### Clean data and change the format for data for Weather ###################################################################################
import pandas as pd
Month_change = ['January','February','March','April','May','June','July','August','September','October','November','December']
Final = pd.read_csv('TreeRecovery_WeatherFinal.csv')
Month_list = []
Month_number_list=[]
Day_list =[]
Year_list=[]
ChangeDate = []

def changeMonthDay(i):
    if len(i)>1:
        return i
    else:
        return('0'+ i)
    
for row in range(len(Final)):
    Date = Final['WDate'][row].split('/')
    Year = '20'+ Date[2]
    ChangeDate.append(Year + '-' + changeMonthDay(Date[0])+ '-' + changeMonthDay(Date[1]))
    Day = Date[1]
    Month = Month_change[int(Date[0])-1]
    Month_Number = Date[0]
    Month_list.append(Month)
    Month_number_list.append(Month_Number)
    Day_list.append(Day)
    Year_list.append(Year)

Final['Year'] = Year_list
Final['Month'] = Month_list
Final['Month_Number'] = Month_number_list
Final['Day'] = Day_list
Final['Date'] = ChangeDate

del Final['Unnamed: 0']
del Final['WeatherID']
del Final['WDate']
Final.to_csv("CIS4400_WeatherFinal.csv")


# In[54]:


################################### Clean data and change the format for data for 311 ###################################################################################
df311 = pd.read_csv('TreeRecovery.csv')
df311


# In[55]:


df311 = df311.dropna()
df311 = df311.reset_index(drop=True)


# In[56]:


Month_change = ['January','February','March','April','May','June','July','August','September','October','November','December']
Month_list = []
Month_number_list=[]
Day_list =[]
Year_list=[]
Created_Date = []
Closed_Date = []

for row in range(len(df311)):
    Date = df311['created_date'][row].split('T')
    Created_Date.append(Date[0])
    Date = Date[0].split('-') 
    Year = Date[0]
    Year_list.append(Year)
    
    if str(Date[2])[0] == '0':
        Day = str(Date[2])[1]
        Day_list.append(Day)

    else:
        Day = Date[2]
        Day_list.append(Day)
    
    if str(Date[1])[0] == '0':
        Month_Number = str(Date[1])[1]
        Month = Month_change[int(Month_Number)-1]
        Month_list.append(Month)
        Month_number_list.append(Month_Number)
    else:
        Month_Number = Date[1]
        Month = Month_change[int(Month_Number)-1]
        Month_list.append(Month)
        Month_number_list.append(Month_Number)

df311['Created_Year'] = Year_list
df311['Created_Month'] = Month_list
df311['Created_Month_Number'] = Month_number_list
df311['Created_Day'] = Day_list
df311['Created_Date'] = Created_Date


# In[57]:


df311


# In[58]:


################################### Clean data and change the format for data for 311 ###################################################################################
Month_list = []
Month_number_list=[]
Day_list =[]
Year_list=[]

for row in range(len(df311)):
    Date = df311['closed_date'][row].split('T')
    Closed_Date.append(Date[0])
    Date = Date[0].split('-') 
    Year = Date[0]
    Year_list.append(Year)
    if str(Date[2])[0] == '0':
        Day = str(Date[2])[1]
        Day_list.append(Day)
    else:
        Day = Date[2]
        Day_list.append(Day)
    
    if str(Date[1])[0] == '0':
        Month_Number = str(Date[1])[1]
        Month = Month_change[int(Month_Number)-1]
        Month_list.append(Month)
        Month_number_list.append(Month_Number)
    else:
        Month_Number = Date[1]
        Month = Month_change[int(Month_Number)-1]
        Month_list.append(Month)
        Month_number_list.append(Month_Number)
df311['Closed_Year'] = Year_list
df311['Closed_Month'] = Month_list
df311['Closed_Month_Number'] = Month_number_list
df311['Closed_Day'] = Day_list
df311['Closed_Date'] = Closed_Date


# In[59]:


df311


# In[60]:


del df311['Unnamed: 0']
del df311['created_date']
del df311['closed_date']


# In[61]:


############################################################ Fix the capital letter ###################################################################################


#Function to capitalize the first letter of each word
def changecap(words):
    final_word = ''
    if len(words)>1:
        words_list = words.split(' ')
        for word in words_list:
            word = word.capitalize()
            final_word += word
            final_word += '_'
        return final_word[:-1]
    else:
        return word

    
newcity = []
newborough = []
state = []
for row in range(len(df311)):
    newcity_name = changecap(df311['city'][row])
    newborough_name = changecap(df311['borough'][row])
    state.append('NY')
    newcity.append(newcity_name)
    newborough.append(newborough_name)

df311['city'] = newcity
df311['borough'] = newborough  
df311['state'] = state


# In[62]:


df311.to_csv("311Final.csv")


# In[ ]:




