import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import math
from scipy.interpolate import interp1d
import requests
import io

url = "https://raw.githubusercontent.com/Alexandros-Korolis/python_weather_project/main/weather_data.csv" 
download = requests.get(url).content

data = pd.read_csv(io.StringIO(download.decode('utf-8')))
df=pd.DataFrame(data)

#Απάντηση στο ερώτημα 1 .

df.loc[17,["HIGH"]]=np.nan
df.loc[61,["HIGH"]]=np.nan
df.loc[55,["LOW"]]=np.nan
df.loc[103,["LOW"]]=np.nan
df1=df.loc[14:20,["HIGH"]]
df1new=df1.astype(float).interpolate(method="cubic")
df.loc[17,["HIGH"]]=df1new.iloc[3,0]
df2=df.loc[58:64,["HIGH"]]
df2new=df2.astype(float).interpolate(method="cubic")
df.loc[61,["HIGH"]]=df2new.iloc[3,0]
df3=df.loc[52:58,["LOW"]]
df3new=df3.astype(float).interpolate(method="cubic")
df.loc[55,["LOW"]]=df3new.iloc[3,0]
df4=df.loc[100:106,["LOW"]]
df4new=df4.astype(float).interpolate(method="cubic")
df.loc[103,["LOW"]]=df4new.iloc[3,0]
df.loc[334:364,["MONTH"]]="DEC"

#Απάντηση στο ερώτημα 2 .

df.loc[365,["HIGH"]]=df["HIGH"].astype(float).max()
df.loc[365,["LOW"]]=abs(df["LOW"].astype(float).min())
df.loc[365,["WINDHIGH"]]=abs(df["WINDHIGH"].min())
df.loc[365,["TEMP"]]=df["TEMP"].mean()
df.loc[365,["RAIN"]]=df["RAIN"].sum()
df.loc[365,["HDD"]]=df["HDD"].sum()
df.loc[365,["CDD"]]=df["CDD"].sum()

#Απάντηση στο ερώτημα 3 .
diam=df["TEMP"].median()
typkap=df["TEMP"].std()
print("Η διάμεσος των μέσων θερμοκρασιών είναι : ",diam)
print("Η τυπική απόκλιση των μέσων θερμοκρασιών είναι : ",typkap)

#Απάντηση στο ερώτημα 4 .
for i in ['N', 'W', 'SW', 'SSW', 'S', 'NW', 'WNW', 'WSW', 'E', 'SE', 'ESE',
       'NE', 'ENE', 'NNW', 'SSE']:
    print(i ,(df['DIR']== i).sum())
    
wdir=['N', 'W', 'SW', 'SSW', 'S', 'NW', 'WNW', 'WSW', 'E', 'SE', 'ESE',
       'NE', 'ENE', 'NNW', 'SSE']
ndays=[103,19,28,22,31,14,20,17,5,65,15,4,2,9,11]
plt.pie(ndays,labels=wdir,shadow=False,autopct='%1.1f%%')

#Απάντηση στο ερώτημα 5 .

dfht=df.loc[:365,["HIGH","TIME"]]
xhigh=dfht.groupby('TIME')
yhigh=xhigh.count()
xmeg=yhigh['HIGH'].max()
print("Η ζητούμενη ώρα και συχνότητα για τις μέγιστες θερμοκρασίες είναι : " ,yhigh.loc[yhigh["HIGH"]==xmeg])
dflt=df.loc[:365,["LOW","TIME.1"]]
xlow=dflt.groupby("TIME.1")
ylow=xlow.count()
xelax=ylow["LOW"].max()
print("Η ζητούμενη ώρα και συχνότητα για τις ελάχιστες θερμοκρασίες είναι : " ,ylow.loc[ylow["LOW"]==xelax])

#Απάντηση στο ερώτημα 6 .
dfask6=df.loc[:364,["MONTH","DAY","HIGH","LOW"]]
l=list(range(0,365))
for i in range (0,365):
    l[i]=dfask6.loc[i,["HIGH","LOW"]].astype(float).var()
dfask6['Διακύμανση']=l
megdiak=dfask6["Διακύμανση"].max()
print(dfask6.loc[dfask6["Διακύμανση"]==megdiak])

#Απάντηση στο ερώτημα 7 .
dfask7=df.loc[:365,["DAY","DIR"]].groupby('DIR').count()
permeres=dfask7['DAY'].max()
print(dfask7.loc[dfask7['DAY']==permeres])

#Απάντηση στο ερώτημα 8 .
dfask8=df.loc[:365,["DIR","WINDHIGH"]]
mwspeed=dfask8["WINDHIGH"].max()
print(dfask8.loc[dfask8["WINDHIGH"]==mwspeed])

#Απάντηση στο ερώτημα 9 .
dfask9=df.loc[:364,["DIR","TEMP"]]    
x9=dfask9.groupby('DIR').mean() 
thermmeg=x9['TEMP'].max()
thermmin=x9['TEMP'].min()
print('Η μέση θερμοκρασία για κάθε διεύθυνση του ανέμου είναι :',x9)
print('Η διεύθυνση με την μεγαλύτερη μέση θερμοκρασία είναι :', x9.loc[x9['TEMP']==thermmeg])   
print('Η διεύθυνση με την μικρότερη μέση θερμοκρασία είναι :', x9.loc[x9['TEMP']==thermmin])

#Απάντηση στο ερώτημα 10 . 
xask10=df.loc[:364,['MONTH','RAIN']].groupby('MONTH').sum()   
xask10.plot.bar()

#Απάντηση στο ερώτημα 11 .
df.loc[334:364,['DAY','MONTH','TEMP']]
imerom=list(range(1,32))
thermo=[17.1,17.4,17.3,13.6,9.3,7,9.4,11.4,12.9,12.9,10.7,12.4,14.4,14.4,13.4,15.3,14.4,10.5,7.2,6.7,5.9,4.6,5.4,6.4,10.9,11.1,11.9,13.0,11.3,9.9,8.9]
plt.scatter(imerom,thermo)
I=np.ones((31,),dtype='float')
A=np.c_[imerom, I]
np.linalg.lstsq(A,thermo,rcond='warn')
xp=np.arange(1,32.01,0.01)
yp=-0.18048387*xp+14.08129032
ax=plt.subplot(1,1,1)
plt.title('least squares')
ax.scatter(imerom,thermo)
ax.plot(xp,yp)
plt.show()  
ynew=-0.18048387*25+14.08129032
print('Η ζητούμενη θερμοκρασία θα είναι :',ynew)

#Απάντηση στο ερώτημα 12 .
df.loc[0:59,["TEMP","HIGH","LOW"]].astype(float).plot(xlabel="Χειμώνας")
df.loc[59:150,["TEMP","HIGH","LOW"]].astype(float).plot(xlabel="Άνοιξη")
df.loc[151:242,["TEMP","HIGH","LOW"]].astype(float).plot(xlabel="Καλοκαίρι")
df.loc[152:364,["TEMP","HIGH","LOW"]].astype(float).plot(xlabel="Φθνινόπωρο")


#Απάντηση στο ερώτημα 13 .
def vrox(athr):
    if athr < 400 :
        print("Λειψυδρία")
    elif athr >=400 and athr <600:
        print("Ικανοποιητικά ποσά βροχής")
    elif athr >= 600:
        print("Υπερβολική βροχόπτωση")
