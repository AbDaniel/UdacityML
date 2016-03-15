#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pandas as pd
import numpy as np

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

df = pd.DataFrame(enron_data).T

# print len(df[df['poi'] == True])
print df.columns

print df.loc['PRENTICE JAMES']['total_stock_value']
print df.loc['COLWELL WESLEY']['from_this_person_to_poi']
print df.loc['SKILLING JEFFREY K']['exercised_stock_options']
print df.loc['SKILLING JEFFREY K']['total_payments']
print df.loc['LAY KENNETH L']['total_payments']
print df.loc['FASTOW ANDREW S']['total_payments']
print len(df['salary'][df['salary'] != 'NaN'])
print len(df['email_address'][df['email_address'] != 'NaN'])
print len(df['total_payments'][df['total_payments'] != 'NaN'])
print df['total_payments'][df['total_payments'] != 'NaN']
print len(df[ (df['poi'] == True) & (df['total_payments'] != 'NaN') ])
print len(df[ (df['poi'] == True) ])