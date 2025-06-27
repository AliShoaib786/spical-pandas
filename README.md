# spical-pandas
ais ma spical  pandas ka seraf code haa only on  







#import pandas as pd
#
# pf = pd.read_csv('imdb-top-1000.csv')
#
#
# # print(pf.columns)
#
# numeric_cols = ['IMDB_Rating', 'No_of_Votes', 'Gross']
#
# genre = pf.groupby(['Director', 'Star1'])[numeric_cols]
#
# print(genre.agg(['min', 'max', 'mean']))
from operator import concat, index
from tokenize import group

import numpy
# ==================================  to find top ten players the maximum runs in ipl                   ==========================

# import pandas as pd
# pf=pd.read_csv('deliveries.csv')
# ipl=pf.groupby('batsman')
# print(ipl['batsman_runs'].sum().sort_values(ascending=False).head(10))

# ================================  find the batsman with max no of six  ====================
#
# import pandas as pd
# pf=pd.read_csv('deliveries.csv')
# # print(pf.info())
# # ipl=pf.groupby('batsman')
# six=pf[pf['batsman_runs'] == 6]
# ipl=six.groupby('batsman')
# print(ipl['batsman'].count().sort_values(ascending=False).head(1))


# ===============================  in ipl the last five overs which most player 4 and 6  =====================

# import pandas as pd
#
# pf = pd.read_csv('deliveries.csv')
#
# temp = pf[pf['over'] > 15]
#
#
# six = pf[(pf['batsman_runs'] == 6) | (pf['batsman_runs'] == 4)]
#
#
# ipl = six.groupby('batsman')
# print(ipl['batsman_runs'].count().sort_values(ascending=False))


# ====================    find Varat kohli record against  all team   =============================


#
# import pandas as pd
#
# pf = pd.read_csv('deliveries.csv')
# # print(pf.info())
#
# ipl=pf[pf['batsman']=='V Kohli']
# temp=ipl.groupby('bowling_team')
# print(temp['batsman_runs'].sum())
# print(temp)




# ========================    pandas   vedioes  no 6   total = 60   ====================================
#
# import pandas as pd
# # student1 = pd.read_csv('students_pandas_6.csv')
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
# # print(pd.concat([november,december],ignore_index=True))
#
# multi=pd.concat([november,december],keys=['november','december'])
#
# print(multi.loc['december'])
#
# print(multi.loc[('december',1)])


# ===================  i concat the data_fram in horizotal  =====================

#
# import pandas as pd
# student1 = pd.read_csv('students_pandas_6.csv')
#
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
# reg=pd.concat([november,december],ignore_index=True))



# ===============================   iner join  =================

#
# import pandas as pd
# student1 = pd.read_csv('students_pandas_6.csv')
#
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
# print(student1.merge(december,how='inner',on='student_id'))



#   ======================    left  join  ===========

#
# import pandas as pd
# student1 = pd.read_csv('students_pandas_6.csv')
# coures=pd.read_csv('courses_pandas_6.csv')
#
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
# print(december.merge(coures,how='left',on='course_id'))


# ====================   right join  ======================================


# import pandas as pd
# from pandas import concat
#
# student1 = pd.read_csv('students_pandas_6.csv')
# coures=pd.read_csv('courses_pandas_6.csv')
#
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
#
# temp=pd.DataFrame({
#     'student_id':[26,27,28],
#     'name':['shoaib','hamza','asad'],
#     'partner':[28,26,17]
# })
# students=concat([student1,temp],ignore_index=True)
# print(students)
# print(december)
#
# print(students.merge(december,how='left',on='student_id').tail(10))


# =====================================   outer  join   ==========================

# import pandas as pd
# from pandas import concat
#
# student1 = pd.read_csv('students_pandas_6.csv')
# coures=pd.read_csv('courses_pandas_6.csv')
#
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
#
# temp=pd.DataFrame({
#     'student_id':[26,27,28],
#     'name':['shoaib','hamza','asad'],
#     'partner':[28,26,17]
# })
# students=concat([student1,temp],ignore_index=True)
# print(students)
# print(december)
#
# print(students.merge(december,how='outer',on='student_id').tail(10))



# ================================    find the total revenue of the company  ========================
#
# import pandas as pd
# from pandas import concat
#
# student1 = pd.read_csv('students_pandas_6.csv')
# coures=pd.read_csv('courses_pandas_6.csv')
#
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
# print(coures)
# registeration=concat([december,november],ignore_index=True)
#
# print(registeration.merge(coures,how='inner',on='course_id')['price'].sum())



# ====================  fnd the  revenue month by mnth  ========================



# import pandas as pd
#
#
#
# student1 = pd.read_csv('students_pandas_6.csv')
# coures=pd.read_csv('courses_pandas_6.csv')
#
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
#
# registeration=pd.concat([december,november],keys=['december','november']).reset_index()
# # print(registeration)
# print(registeration.merge(coures,on='course_id').groupby('level_0')['price'].sum())


 # =====================   find print  the colunms  name , course_name ,prices   ================

# import pandas as pd
# from pandas import concat
#
# student1 = pd.read_csv('students_pandas_6.csv')
# coures=pd.read_csv('courses_pandas_6.csv')
#
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
# print(coures)
# print(student1)
#
# register=concat([november,december],ignore_index=True)
# print(register)
# merge=register.merge(coures,on='course_id').merge(student1,on='student_id')
# print(merge[['name','course_name','price']])




# ========================  drow the plot on the base of each course revenue  ============================
# import matplotlib.pyplot as plt
# import pandas as pd
# from pandas import concat
#
# coures=pd.read_csv('courses_pandas_6.csv')
# # print(coures)
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
# merge=concat([november,december],ignore_index=True)
# group=merge.merge(coures,on='course_id')
# print(group.groupby('course_name')['price'].sum().plot(kind='bar'))
# plt.show()



# ==========================  find the student that enrolled  both month  ===================
# import numpy as np
# import pandas as pd
#
#
# coures=pd.read_csv('courses_pandas_6.csv')
# student1 = pd.read_csv('students_pandas_6.csv')
# # print(student1)
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
# intersect=np.intersect1d(november['student_id'],december['student_id'])
#
# print(student1[student1['student_id'].isin(intersect)])


#  =========================    find the course that donot enroll the student  ========================

# import numpy as np
# import pandas as pd
#
#
#
# coures=pd.read_csv('courses_pandas_6.csv')
#
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
# concatenate=pd.concat([november,december],ignore_index=True)
# setdiff=np.setdiff1d(coures['course_id'],concatenate['course_id'])
# print(coures[coures['course_id'].isin(setdiff)])


#  ============================   find the student how donot enrolled  any coures  ======================

# import numpy as np
# import pandas as pd




#
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
# student1 = pd.read_csv('students_pandas_6.csv')
#
# concatenate=pd.concat([november,december],ignore_index=True)
# setdiffer=np.setdiff1d(student1['student_id'],concatenate['student_id'])
# print(student1[student1['student_id'].isin(setdiffer)])




# =================  print the student name and partner name   from all the enrolled student================
#
# import numpy as np
# import pandas as pd
#
#
# student1 = pd.read_csv('students_pandas_6.csv')
# print(student1)
#
# print(student1.merge(student1,how='inner',left_on='partner',right_on='student_id')[['name_x','name_y']])



# ===================    find the student how enrolled top most three  ======================
#
# import pandas as pd
#
# student1 = pd.read_csv('students_pandas_6.csv')
# # print(student1)
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
# concatenate=pd.concat([november,december],ignore_index=True)
# print(concatenate.merge(student1,on='student_id').groupby(['student_id','name'])['name'].count().sort_values(ascending=False).head(3))



# ============================   find the top three student who send the memory most   ============================


#
# import pandas as pd
#
# coures=pd.read_csv('courses_pandas_6.csv')
# student1 = pd.read_csv('students_pandas_6.csv')
# november=pd.read_csv('reg-month1_pandas_6.csv')
# december=pd.read_csv('reg-month2_pandas_6.csv')
# concatenate=pd.concat([november,december],ignore_index=True)
#
#
# print(student1.merge(concatenate,on='student_id').merge(coures,on='course_id').groupby(['student_id','name']).sum())


# ==========================   pandas session  7    multindex object   ===========================
#
# import pandas as pd
# value_index=[('punjab',2019),('punjab',2020),('punjab',2021),('punjab',2022),('punjabi',2019),('punjabi',2020),('punjabi',2021),('punjabi',2022)]
# a=pd.MultiIndex.from_tuples(value_index)
# print(a.levels)


# =============================   the second method of  MultiIndex.from_tuples  =============================
#
# import pandas as pd
# value_index=[('punjab',2019),('punjab',2020),('punjab',2021),('punjab',2022),('punjabi',2019),('punjabi',2020),('punjabi',2021),('punjabi',2022)]
# # a=pd.MultiIndex.from_product([['punjab','punjabi'],[2019,2020,2021,2022]])
# a = pd.MultiIndex.from_tuples(value_index)
# s=pd.Series([1,2,3,4,5,6,7,8],a)
# print(s)
# print(s[('punjab',2019)])



#    ======================  2 dimension series convert the data_frame throught unstrack function  ==================
#
# import pandas as pd
# value_index=[('punjab',2019),('punjab',2020),('punjab',2021),('punjab',2022),('punjabi',2019),('punjabi',2020),('punjabi',2021),('punjabi',2022)]
# # a=pd.MultiIndex.from_product([['punjab','punjabi'],[2019,2020,2021,2022]])
# a = pd.MultiIndex.from_tuples(value_index)
# s=pd.Series([1,2,3,4,5,6,7,8],a)
# print(s.unstack())




#    ======================  data frame convert the multi index series throught strack function  ==================
#
# import pandas as pd
# value_index=[('punjab',2019),('punjab',2020),('punjab',2021),('punjab',2022),('punjabi',2019),('punjabi',2020),('punjabi',2021),('punjabi',2022)]
# # a=pd.MultiIndex.from_product([['punjab','punjabi'],[2019,2020,2021,2022]])
# a = pd.MultiIndex.from_tuples(value_index)
# s=pd.Series([1,2,3,4,5,6,7,8],a)
# test=s.unstack()
# print(test.stack())



# =========================   how to create the multi index  in  dataframe  ======================
#
# import pandas as pd
# a=pd.MultiIndex.from_product([['punjab','punjabi'],[2019,2020,2021,2022]])
#
# fd=pd.DataFrame(
#     [
#        [2,4],
#        [3,5],
#        [6,7],
#        [1,4],
#        [2,1],
#        [1,9],
#        [6,9],
#        [5,6]
#     ],
#     a,columns=['average','student']
# )
#
# print(fd)
# print(fd['average'])
#

# ==================   how to create multiIndex dataframe  from columns prespective  ===============

#
# import pandas as pd
#
#
#
#
# fd=pd.DataFrame(
#     [
#        [2,4,6,2],
#        [3,5,5,6],
#        [6,7,10,11],
#        [1,4,21,45],
#        [2,1,9,5],
#        [1,9,8,5],
#        [6,9,12,7],
#        [5,6,13,67]
#     ],
#     index=[2019,2020,2021,2022,2023,2024,2025,2026],
#     columns=pd.MultiIndex.from_product([['punjab','punjabi'],['average','student']])
# )
#
# print(fd)
# print(fd['punjabi'])
#
# print(fd['punjabi']['student'])
# # =========  if you fetch any row
# print(fd.loc[2019])
#


# ====================  we create the multi  columns and multi index  in the data frame  ===================

#
# import pandas as pd
#
#
# a=pd.MultiIndex.from_product([['punjab','punjabi'],[2019,2020,2021,2022]])
#
# fd=pd.DataFrame(
#     [
#        [2,4,6,2],
#        [3,5,5,6],
#        [6,7,10,11],
#        [1,4,21,45],
#        [2,1,9,5],
#        [1,9,8,5],
#        [6,9,12,7],
#        [5,6,13,67]
#     ],
#     index=a,
#     columns=pd.MultiIndex.from_product([['lahore','Islamabad'],['average','student']])
# )
# print(fd)
#
# print(fd.loc['punjab', ('lahore', 'student')])
#
# print(fd.loc[('punjab', 2019), ('lahore', 'average')])



# ================================  unstack  function  ======================================


# import pandas as pd
# a=pd.MultiIndex.from_product([['punjab','punjabi'],[2019,2020,2021,2022]])
#
# fd=pd.DataFrame(
#     [
#        [2,4],
#        [3,5],
#        [6,7],
#        [1,4],
#        [2,1],
#        [1,9],
#        [6,9],
#        [5,6]
#     ],
#     a,columns=['average','student']
# )
# print(fd)
# test=fd.unstack()
# print(test)
#
# again=test.unstack()
# print(again)


#
# # ================================   stack function =======================
#
# import pandas as pd
# a=pd.MultiIndex.from_product([['punjab','punjabi'],[2019,2020,2021,2022]])
#
# fd=pd.DataFrame(
#     [
#        [2,4],
#        [3,5],
#        [6,7],
#        [1,4],
#        [2,1],
#        [1,9],
#        [6,9],
#        [5,6]
#     ],
#     a,columns=['average','student']
# )
# # print(fd.unstack().stack().stack())
#
# print(fd.stack())


# =======================  unstack()   ========================


#
# import pandas as pd
#
#
# a=pd.MultiIndex.from_product([['punjab','punjabi'],[2019,2020,2021,2022]])
#
# fd=pd.DataFrame(
#     [
#        [2,4,6,2],
#        [3,5,5,6],
#        [6,7,10,11],
#        [1,4,21,45],
#        [2,1,9,5],
#        [1,9,8,5],
#        [6,9,12,7],
#        [5,6,13,67]
#     ],
#     index=a,
#     columns=pd.MultiIndex.from_product([['lahore','Islamabad'],['average','student']])
# )
# print(fd)
# ==================   this is un strack   ================================
# print(fd.unstack().unstack())

#   =============================    this is stack()================

#print(fd.stack())


# ========================   that put  the data frame method  are apply  ==============

#
# import pandas as pd
#
#
# a=pd.MultiIndex.from_product([['punjab','punjabi'],[2019,2020,2021,2022]])
#
# fd=pd.DataFrame(
#     [
#        [2,4,6,2],
#        [3,5,5,6],
#        [6,7,10,11],
#        [1,4,21,45],
#        [2,1,9,5],
#        [1,9,8,5],
#        [6,9,12,7],
#        [5,6,13,67]
#     ],
#     index=a,
#     columns=pd.MultiIndex.from_product([['lahore','Islamabad'],['average','student']])
# )
# print(fd)
#
# # ======================   that show first five row =========================
# print(fd.head())
#
# # ==========================    that show shape of the data frame  ===============
# print(fd.shape)
#
# # =====================   that show all the name of columns and index nmame  =============
# print(fd.info())
# # ===========================  that show the duplicate values  ==========================
#
# print(fd.duplicated())
#
# # =============================  that check the null values in dtaframe  ============
#
# print(fd.isnull)



# ==================extracting the row and column   from 4D multi indexing object  ============================



#
#
#
# import pandas as pd
#
#
# a=pd.MultiIndex.from_product([['punjab','punjabi'],[2019,2020,2021,2022]])
#
# fd=pd.DataFrame(
#     [
#        [2,4,6,2],
#        [3,5,5,6],
#        [6,7,10,11],
#        [1,4,21,45],
#        [2,1,9,5],
#        [1,9,8,5],
#        [6,9,12,7],
#        [5,6,13,67]
#     ],
#     index=a,
#     columns=pd.MultiIndex.from_product([['lahore','Islamabad'],['average','student']])
# )
# print(fd)
# # =======================  that use for the row
# print(fd.loc[('punjabi',2019)])
# # =========================   that   use for the columns
# print(fd.loc['punjab',('lahore','student')])
# # ===================  that use for  the alternative rows
# print(fd.loc[('punjab',2019):('punjabi',2020):2])
#
# print(fd.iloc[0:5:2])
#
# # ====================================   that extracting the colum  from the multi Indexing object  ============
#
# # ===============================  that fetch the  all the columns of the islamabad  ============
# print(fd['Islamabad'])
#
# # =======================================  that  fetch the columns of student in islamabad  ==================
#
# print(fd['Islamabad']['student'])
#
# #  ==============================  that fetch the alternative columns   ===========================
#
# print(fd.iloc[:,1:3])
#
# #===================== that  show the slices   of multi indexing object  =============
#
# print(fd.iloc[[0,4],[1,2]])
#




# ========================   sort indexing  on multi_index_object  ==============================


#
#
# import pandas as pd
#
#
# a=pd.MultiIndex.from_product([['punjab','punjabi'],[2019,2020,2021,2022]])
#
# fd=pd.DataFrame(
#     [
#        [2,4,6,2],
#        [3,5,5,6],
#        [6,7,10,11],
#        [1,4,21,45],
#        [2,1,9,5],
#        [1,9,8,5],
#        [6,9,12,7],
#        [5,6,13,67]
#     ],
#     index=a,
#     columns=pd.MultiIndex.from_product([['lahore','Islamabad'],['average','student']])
# )
# print(fd)
#
# #   ===============================    sort_ indexing  ===================
#
# print(fd.sort_index(ascending=[False,True]))
#
#
# # =========================transposed  the indexing   ========================
#
# print(fd.transpose())
#
# # ==========================  swaplevel   ====================================
#
# print(fd.swaplevel(axis=1))


# ================================  melt  ====================================
#
# import pandas as pd
#
#
# pf=pd.DataFrame({'cse':[12],'ics':[120],'icom':[130]})
# print(pf)
# print(pf.melt(var_name='branch',value_name='no_of_student'))

# =======================  melt another example ==================================
#
# import pandas as pd
# pf=pd.DataFrame({
#     'branch':['ics','icom','Fsc'],
#     '2020':[120,130,140],
#     '2021':[150,160,170],
#     '2022':[100,120,140]
# })
# print(pf)
# print(pf.melt())
# print(pf.melt(id_vars='branch',var_name='year',value_name='no_of_student'))
#
#
# import pandas as pd
#
# # Load death data (actually confirmed cases due to file name)
# death = pd.read_csv('time_series_covid19_confirmed_global.csv')
# deaths = death.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
#                     var_name='date', value_name='confirmed')
#
# # Load confirmed data (actually deaths due to file name)
# confirm = pd.read_csv('time_series_covid19_deaths_global.csv')
# confirms = confirm.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
#                         var_name='date', value_name='deaths')
#
# # Merge on all identifying columns including 'date'
# merged = confirms.merge(deaths, on=['Province/State', 'Country/Region', 'Lat', 'Long', 'date'])
#
# merged = confirms.merge(deaths, on=['Province/State', 'Country/Region', 'Lat', 'Long', 'date'])
#
# print(merged['Province/State'])
#
#


import numpy as np
import pandas as pd
# import seaborn as sns
# df=sns.load_dataset('tips')
# print(df.head())
# #
# print(df.groupby('sex')[['total_bill']].mean())
#
# print(df.groupby(['sex','smoker'])[['total_bill']].mean().unstack())
#
# print(df.pivot_table(index='sex',columns='smoker',values='total_bill',aggfunc='count'))


# =======================  multi demension in pivot table ==============================

# import seaborn as sns
# df=sns.load_dataset('tips')
# print(df.head())
# print(df.pivot_table(index=['sex','smoker'],columns=['day','time'],aggfunc={'size':'mean','tip':'max','total_bill':'sum'}))


# =========================  margin calculate the sum in row and columnwise  =========================
#
# import seaborn as sns
# df=sns.load_dataset('tips')
# print(df.head())
# print(df.pivot_table(index='sex',columns='smoker',values='total_bill',margins=True))
#
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# pf=pd.read_csv('expense_data.csv')
# print(pf.info())
# pf['Date']=pd.to_datetime(pf['Date'])
# print(pf.info())
#
# pf['month']=pf['Date'].dt.month_name()
# # print(pf.head())
# print(pf['Category'])
#
# print(pf.pivot_table(index='month',columns='Category',values='INR',aggfunc='mean').plot())
#
# plt.show()



#======================================    vectorized  string operation  =============================
#
# import pandas as pd
# import numpy as np
# s=['cat','mat',None,'rat']
# p=(i.startswith('c') for i in s)
# print(list(p))


#========================================  that handl this problem throught  pandas   ==================

# import pandas as pd
# import numpy as np
# s=pd.Series(['cat','mat',None,'rat'])
# print(s.str.startswith('c'))


# # import pandas as pd
# # import numpy as np
#
# pf=pd.read_csv('titanic.csv')

#    ==========================  all the name in lower case   =============
# print(pf['Name'].str.lower())


# ==============================  show the upper case  all name

# print(pf['Name'].str.upper())



# ==============================  show the captilized case  all name

# print(pf['Name'].str.capitalize())



# ==============================  show the title case  all name

# print(pf['Name'].str.title())


# ==================   find the name how is max lenght  in this name columns  ===============


import pandas as pd
import numpy as np

# pf=pd.read_csv('titanic.csv')
# print(pf['Name'][pf['Name'].str.len() == 82].values[0])



#  ================================ we create the three colum fist name sir name and title name drive from colum name
#
# import pandas as pd
# import numpy as np
# from PIL.ImageOps import expand
# from fontTools.merge.util import first
#
# pf=pd.read_csv('titanic.csv')
#
# pf['last_name']=pf['Name'].str.split(',').str.get(0)
# print(pf['last_name'])
#
# pf[['tilte_name','first_name']]=pf['Name'].str.split(',').str.get(1).str.strip().str.split(' ',n=1, expand=True)
# print(pf[['tilte_name','first_name']])
#
#
#
# print(pf.head())


#
#
# import pandas as pd
# import numpy as np

# pf = pd.read_csv('titanic.csv')

# Extract last name
# pf['last_name'] = pf['Name'].str.split(',').str.get(0)

# Extract title and first name into two new columns
# pf[['title_name', 'first_name']] = pf['Name'].str.split(',').str.get(1).str.strip().str.split(' ', n=1, expand=True)
#
# # Print the new columns
# print(pf[['title_name', 'first_name','last_name']])
# # print(pf['last_name'])
# print(pf['title_name'].value_counts())

# pf['title_name']=pf['title_name'].str.replace('Mlle.','Miss.')
# print(pf['title_name'].value_counts())





#    ===================   filltering  startswith and the endswith  =======================

#
# print(pf[pf['first_name'].str.endswith('A')])

# =================    check  this in the first_name colum  the name are start with digit  ==

# print(pf[pf['first_name'].str.isdigit()])


# =================================   check  that jhon in the each  colum row in first_nname  ============

# print(pf[pf['first_name'].str.contains('john', case=False)])





# ==================  we fetch the last name start with vowel word and end with vowel word  are fetch with those name====



#
#
# import pandas as pd
# import numpy as np
# pf = pd.read_csv('titanic.csv')
#
# pf['last_name'] = pf['Name'].str.split(',').str.get(0)
#
# pf[['title_name', 'first_name']] = pf['Name'].str.split(',').str.get(1).str.strip().str.split(' ', n=1, expand=True)
#
# print([['title_name','last_name','first_name']])
#
# print(pf.head())
# print(pf[pf['first_name'].str.contains('^[aeiouAEIOU].+[aeiouAEIOU]$')])
#
# print(pf[pf['first_name'].str.contains('^[^aeiouAEIOU].+[^aeiouAEIOU]$')])
#
#
#
#



# =======================   slicing on any columns  ==================================

#
#
# import pandas as pd
# import numpy as np
# pf = pd.read_csv('titanic.csv')
#
# print(pf['Name'].str[::-1])




# =====================================   create the timestamp  ============
#
# import pandas as pd
# pf=pd.Timestamp('2022')
# print(pf)
#
# pff=pd.Timestamp('17 may 2025 9:30')
# print(pff)


# ===================================  time in python  ====================
# import pandas as pd
# import datetime as dt
#
# from pandas import Timestamp
#
# df=Timestamp(dt.datetime(2025,5,17,11,9,44))
# print(df)
# print(df.year)
# print(df.month)
# print(df.day)
# print(df.hour)
# print(df.minute)
# print(df.second)


# ================   datetime object n numpy   ======================
#
# import numpy as np
# date=np.array('2025-05-17',dtype=np.datetime64)
# print(date)
#
# print(date+np.arange(12))




# ===================   Datetime index to create  ==============================
# import pandas as pd
# pf=pd.DatetimeIndex(['2022/04/12','2023/09/17','2024/08/03'])
# print(pf[2])


# =============================  datetime in python  =========================
#
# import pandas as pd
# import datetime as dt
#
# from pandas import DatetimeIndex
#
# pf=pd.DatetimeIndex([dt.datetime(2023,3,4),dt.datetime(2024,5,18),dt.datetime(2025,6,24)])
# print(pd.Series([1,2,3],index=pf))




# ============================== Date_arange_function  ==============================
#
# import pandas as pd
# import datetime as dt
#
# pf=pd.date_range(start='2025/02/01',end='2025/02/28',freq='W-THU')
# print(pf)


# ======================   to find the mnth
# import pandas as pd
# import datetime as dt
#
# pf=pd.date_range(start='2025/02/17',periods=25,freq='M')
# print(pf)



# ===========================  to find which month  is start  ===========================
#
# import pandas as pd
# import datetime as dt
#
# pf=pd.date_range(start='2025/02/17',end='2025/08/12',freq='ME')
# print(pf)
#



# ========================  to_datetime function  ==========================

#
# import pandas as pd
# import datetime as dt
#
# pf=pd.Series(['2023/01/17','2023/07/26','2025/09/27'])
# pff=pd.to_datetime(pf)
# print(pff.dt.year)


# ============================================   example  ============================================
# import matplotlib.pyplot as plt
# import pandas as pd
# pf=pd.read_csv('expense_data.csv')
# pf['Date']=pd.to_datetime(pf['Date'])
# print(pf['Date'].dt.day_name())
#
# plt.plot(pf['Date'],pf['INR'])
# plt.show()
