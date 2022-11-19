#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tkinter import *
from tkinter import simpledialog
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
main_win = Tk()


# In[2]:


data=pd.read_csv('food.csv')  #per 100grams of serving 
data


# In[3]:


Breakfastdata=data['Breakfast']
BreakfastdataNumpy=Breakfastdata.to_numpy()

Lunchdata=data['Lunch']
LunchdataNumpy=Lunchdata.to_numpy()
    
Dinnerdata=data['Dinner']
DinnerdataNumpy=Dinnerdata.to_numpy()

Food_itemsdata=data['Food_items']


# In[4]:


breakfastfoodseparated=[]
Lunchfoodseparated=[]
Dinnerfoodseparated=[]
        
breakfastfoodseparatedID=[]
LunchfoodseparatedID=[]
DinnerfoodseparatedID=[]


# In[5]:


for i in range(len(Breakfastdata)):
    if BreakfastdataNumpy[i]==1:
        breakfastfoodseparated.append( Food_itemsdata[i] )
        breakfastfoodseparatedID.append(i)
    if LunchdataNumpy[i]==1:
        Lunchfoodseparated.append(Food_itemsdata[i])
        LunchfoodseparatedID.append(i)
    if DinnerdataNumpy[i]==1:
        Dinnerfoodseparated.append(Food_itemsdata[i])
        DinnerfoodseparatedID.append(i)


# In[6]:


LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
LunchfoodseparatedIDdata


# In[7]:


val=list(np.arange(5,16))
val


# In[8]:


Valapnd=[0]+[4]+val
Valapnd


# In[9]:


LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.iloc[Valapnd]
LunchfoodseparatedIDdata


# In[10]:


LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
LunchfoodseparatedIDdata


# In[11]:


breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.iloc[Valapnd]
breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
breakfastfoodseparatedIDdata


# In[12]:


DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.iloc[Valapnd]
DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
DinnerfoodseparatedIDdata


# In[13]:


DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.to_numpy()
LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.to_numpy()
breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.to_numpy()
LunchfoodseparatedIDdata


# In[14]:


## K-Means Based  lunch Food
Datacalorie=LunchfoodseparatedIDdata[0:,1:len(LunchfoodseparatedIDdata)]
Datacalorie


# In[15]:


X = np.array(Datacalorie)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)


# In[16]:


XValu=np.arange(0,1+len(kmeans.labels_))
XValu


# In[17]:


lnchlbl=kmeans.labels_
lnchlbl


# In[18]:


## K-Means Based  Dinner Food
Datacalorie=DinnerfoodseparatedIDdata[0:,1:len(DinnerfoodseparatedIDdata)]

X = np.array(Datacalorie)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

XValu=np.arange(0,1+len(kmeans.labels_))
    
# retrieving the labels for dinner food
dnrlbl=kmeans.labels_


# In[19]:


## K-Means Based breakfast Food
Datacalorie=breakfastfoodseparatedIDdata[0:,1:len(breakfastfoodseparatedIDdata)]
    
X = np.array(Datacalorie)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
XValu=np.arange(0,1+len(kmeans.labels_))
    
# retrieving the labels for breakfast food
brklbl=kmeans.labels_


# In[20]:


data_nd=pd.read_csv('nutrition_distribution.csv')
data_nd


# In[21]:


## train set
data1=data_nd.T
data1


# In[22]:


bmicls=[0,1,2,3,4]
agecls=[0,1,2,3,4]


# In[23]:


weightlosscat = data1.iloc[[1,2,7,8]]
weightlosscat


# In[24]:


weightlosscat=weightlosscat.T
weightlosscat


# In[25]:


weightgaincat= data1.iloc[[0,1,2,3,4,7,9,10]]
weightgaincat=weightgaincat.T
weightgaincat


# In[26]:


healthycat= data1.iloc[[1,2,3,4,6,7,9]]
healthycat=healthycat.T
healthycat


# In[27]:


weightlosscat=weightlosscat.to_numpy()
weightgaincat=weightgaincat.to_numpy()
healthycat=healthycat.to_numpy()
weightlosscat


# In[28]:


weightlossfin=np.zeros((len(weightlosscat)*5,6),dtype=np.float32)
weightgainfin=np.zeros((len(weightgaincat)*5,10),dtype=np.float32)
healthyfin=np.zeros((len(healthycat)*5,9),dtype=np.float32)
weightgainfin


# In[29]:


def Weight_Loss():
    ROOT = tk.Tk()
    
    ROOT.withdraw()
    
    USER_INP = simpledialog.askstring(title="Food Timing",
                                      prompt="Enter 1 for Breakfast, 2 for Lunch and 3 for Dinner")
    #print(" Age : %s Years \n Weight: %s Kg \n Hight: %s cm \n Choice: %s \n" % (e1.get(), e2.get(), e3.get(), e4.get()))
    
    age=int(e1.get())
    weight=float(e2.get())
    height=float(e3.get())
    choice=int(e4.get())
    bmi = weight/((height/100)**2) 
    bmi = round(bmi, 1)
    
    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        #print(test_list)
        for i in test_list: 
            if(i == age):
                #print('age is between',str(lp),str(lp+10))
                tr=round(lp/20)  
                #print(tr)
                agecl=round(lp/20)
                #print(agecl)
            
    if ( bmi < 16):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Severely Underweight')
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Underweight')
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Healthy')
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Overweight')
        clbmi=1
    elif ( bmi >=30):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Obesity')
        clbmi=0    
    ti=(bmi+agecl)/2
    
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    
    for zz in range(5):
        for jj in range(len(weightlosscat)):
            valloc=list(weightlosscat[jj])
            #print(valloc)
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t]=np.array(valloc)
            #print(weightlossfin[t])
            yt.append(brklbl[jj])
            #print(yt)
            t+=1
        for jj in range(len(weightlosscat)):
            valloc=list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[r]=np.array(valloc)
            yr.append(lnchlbl[jj])
            r+=1
        for jj in range(len(weightlosscat)):
            valloc=list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[s]=np.array(valloc)
            ys.append(dnrlbl[jj])
            s+=1 
    
    X_test=np.zeros((len(weightlosscat),6),dtype=np.float32)
    
    for jj in range(len(weightlosscat)):
        valloc=list(weightlosscat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti
    print (X_test)
    
    val=int(USER_INP)
    
    if val==1:
        X_train= weightlossfin
        y_train=yt
        
    elif val==2:
        X_train= weightlossfin
        y_train=yr 
        
    elif val==3:
        X_train= weightlossfin
        y_train=ys
        
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
        
    #print ('SUGGESTED FOOD ITEMS ::')
    food=[]
    for ii in range(len(y_pred)):
        if y_pred[ii]==2:
            findata=Food_itemsdata[ii]
            #print(findata)
            food.append(findata)
    food
    root = tk.Tk()
    t = Text(root)
    for x in food:
        t.insert(END, x + '\n')
    t.pack()


# In[30]:


def Weight_Gain():
    ROOT = tk.Tk()
    
    ROOT.withdraw()
    
    USER_INP = simpledialog.askstring(title="Food Timing",
                                      prompt="Enter 1 for Breakfast, 2 for Lunch and 3 for Dinner")
    print(" Age : %s Years \n Weight: %s Kg \n Hight: %s cm \n Choice: %s \n" % (e1.get(), e2.get(), e3.get(), e4.get()))
    
    age=int(e1.get())
    weight=float(e2.get())
    height=float(e3.get())
    choice=int(e4.get())
    bmi = weight/((height/100)**2) 
    bmi = round(bmi, 1)
    
    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        #print(test_list)
        for i in test_list: 
            if(i == age):
                #print('age is between',str(lp),str(lp+10))
                tr=round(lp/20)  
                #print(tr)
                agecl=round(lp/20)
                #print(agecl)
    ti=(bmi+agecl)/2        
    if ( bmi < 16):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Severely Underweight')
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Underweight')
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Healthy')
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Overweight')
        clbmi=1
    elif ( bmi >=30):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Obesity')
        clbmi=0    
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]   
    for zz in range(5):
        for jj in range(len(weightgaincat)):
            valloc=list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[t]=np.array(valloc)
            yt.append(brklbl[jj])
            t+=1
        for jj in range(len(weightgaincat)):
            valloc=list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r]=np.array(valloc)
            yr.append(lnchlbl[jj])
            r+=1
        for jj in range(len(weightgaincat)):
            valloc=list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[s]=np.array(valloc)
            ys.append(dnrlbl[jj])
            s+=1

    
    X_test=np.zeros((len(weightgaincat),10),dtype=np.float32)

   
    for jj in range(len(weightgaincat)):
        valloc=list(weightgaincat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti
    
    val=int(USER_INP)
    
    if val==1:
        X_train= weightgainfin
        y_train=yt
        
    elif val==2:
        X_train= weightgainfin
        y_train=yr 
        
    elif val==3:
        X_train= weightgainfin
        y_train=ys
        
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    
    #print ('SUGGESTED FOOD ITEMS ::')
    food=[]
    for ii in range(len(y_pred)):
        if y_pred[ii]==2:
            findata=Food_itemsdata[ii]
            #print(findata)
            food.append(findata)
    food
    root = tk.Tk()
    t = Text(root)
    for x in food:
        t.insert(END, x + '\n')
    t.pack()


# In[31]:


def Healthy():
    ROOT = tk.Tk()
    
    ROOT.withdraw()
    
    USER_INP = simpledialog.askstring(title="Food Timing",
                                      prompt="Enter 1 for Breakfast, 2 for Lunch and 3 for Dinner")
    print(" Age : %s Years \n Weight: %s Kg \n Hight: %s cm \n Choice: %s \n" % (e1.get(), e2.get(), e3.get(), e4.get()))
    
    age=int(e1.get())
    weight=float(e2.get())
    height=float(e3.get())
    choice=int(e4.get())
    bmi = weight/((height/100)**2) 
    bmi = round(bmi, 1)
    
    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        #print(test_list)
        for i in test_list: 
            if(i == age):
                #print('age is between',str(lp),str(lp+10))
                tr=round(lp/20)  
                #print(tr)
                agecl=round(lp/20)
                #print(agecl)
            
    #print("Your body mass index is: ", bmi)
    if ( bmi < 16):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Severely Underweight')
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Underweight')
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Healthy')
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Overweight')
        clbmi=1
    elif ( bmi >=30):
        messagebox.showinfo('bmi-pythonguides', f'BMI = {bmi} is Obesity')
        clbmi=0    
    ti=(bmi+agecl)/2
    
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    for zz in range(5):
        for jj in range(len(healthycat)):
            valloc=list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthyfin[t]=np.array(valloc)
            yt.append(brklbl[jj])
            t+=1
        for jj in range(len(healthycat)):
            valloc=list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthyfin[r]=np.array(valloc)
            yr.append(lnchlbl[jj])
            r+=1
        for jj in range(len(healthycat)):
            valloc=list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthyfin[s]=np.array(valloc)
            ys.append(dnrlbl[jj])
            s+=1

    X_test=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    
    for jj in range(len(healthycat)):
        valloc=list(healthycat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti
    
    val=int(USER_INP)
    
    if val==1:
        X_train= healthyfin
        y_train=yt
        
    elif val==2:
        X_train= healthyfin
        y_train=yt 
        
    elif val==3:
        X_train= healthyfin
        y_train=ys
        
    clf=RandomForestClassifier(n_estimators=100)  
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    
    #print ('SUGGESTED FOOD ITEMS ::')
    food=[]
    for ii in range(len(y_pred)):
        if y_pred[ii]==2:
            findata=Food_itemsdata[ii]
            #print(findata)
            food.append(findata)
    food
    root = tk.Tk()
    t = Text(root)
    for x in food:
        t.insert(END, x + '\n')
    t.pack()


# In[ ]:


Label(main_win,text="Nutri Health", font='Helvetica 12 bold').grid(row=1,column=1,sticky=W,pady=4)
Label(main_win,text="Nutritiuos Food Suggestion", font='Helvetica 12 bold').grid(row=2,column=1,sticky=W,pady=4)
Label(main_win,text="Age",font='Helvetica 12 bold').grid(row=3,column=0,sticky=W,pady=4)
Label(main_win,text="Weight(in kg)",font='Helvetica 12 bold').grid(row=5,column=0,sticky=W,pady=4)
Label(main_win,text="Height(in cm)", font='Helvetica 12 bold').grid(row=7,column=0,sticky=W,pady=4)
Label(main_win,text="Veg(0)/NonVeg(1)", font='Helvetica 12 bold').grid(row=9,column=0,sticky=W,pady=4)
                                                                       
e1 = Entry(main_win)
e2 = Entry(main_win)                                                                      
e3 = Entry(main_win)
e4 = Entry(main_win)
#e1.focus_force() 

e1.grid(row=3, column=1)
e2.grid(row=5, column=1)                                                                       
e3.grid(row=7, column=1)
e4.grid(row=9, column=1)

Button(main_win,text='Quit',font='Helvetica 8 bold',command=main_win.quit).grid(row=13,column=0,sticky=W,pady=4)
Button(main_win,text='Weight Loss',font='Helvetica 8 bold',command=Weight_Loss).grid(row=11,column=0,sticky=W,pady=4)
Button(main_win,text='Weight Gain',font='Helvetica 8 bold',command=Weight_Gain).grid(row=11,column=1,sticky=W,pady=4)
Button(main_win,text='Healthy',font='Helvetica 8 bold',command=Healthy).grid(row=11,column=2,sticky=W,pady=4)
main_win.geometry("500x500")
main_win.wm_title("DIET RECOMMENDATION SYSTEM")
main_win.configure(bg='light green')
main_win.mainloop()


# In[ ]:




