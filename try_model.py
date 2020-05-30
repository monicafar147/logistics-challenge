"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json
import math

# Machine learning models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None 

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    #feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    #feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #predict_vector = feature_vector_df[['Pickup Lat','Pickup Long',
    #                                    'Destination Lat','Destination Long']]
    feature_vector_df = data
    # Convert time columns to datetime
    feature_vector_df['Pickup - Time'] = pd.to_datetime(feature_vector_df['Pickup - Time']).dt.time
    feature_vector_df['Placement - Time'] = pd.to_datetime(feature_vector_df['Placement - Time']).dt.time
    feature_vector_df['Confirmation - Time'] = pd.to_datetime(feature_vector_df['Confirmation - Time']).dt.time
    feature_vector_df['Arrival at Pickup - Time'] = pd.to_datetime(feature_vector_df['Arrival at Pickup - Time']).dt.time
    feature_vector_df['Arrival at Destination - Time'] = pd.to_datetime(feature_vector_df['Arrival at Destination - Time']).dt.time

    # Converting columns into categories
    feature_vector_df['Vehicle Type'] = feature_vector_df['Vehicle Type'].astype('category')
    feature_vector_df['Platform Type'] = feature_vector_df['Platform Type'].astype('category')
    feature_vector_df['Personal or Business'] = feature_vector_df['Personal or Business'].astype('category')

    feature_vector_df['Placement - Day of Month'] = feature_vector_df['Placement - Day of Month'].astype('category')
    feature_vector_df['Placement - Weekday (Mo = 1)'] = feature_vector_df['Placement - Weekday (Mo = 1)'].astype('category')
    feature_vector_df['Confirmation - Day of Month'] = feature_vector_df['Confirmation - Day of Month'].astype('category')
    feature_vector_df['Confirmation - Weekday (Mo = 1)'] = feature_vector_df['Confirmation - Weekday (Mo = 1)'].astype('category')

    feature_vector_df['Arrival at Pickup - Day of Month'] = feature_vector_df['Arrival at Pickup - Day of Month'].astype('category')
    feature_vector_df['Arrival at Pickup - Weekday (Mo = 1)'] = feature_vector_df['Arrival at Pickup - Weekday (Mo = 1)'].astype('category')
    feature_vector_df['Arrival at Destination - Day of Month'] = feature_vector_df['Arrival at Destination - Day of Month'].astype('category')
    feature_vector_df['Arrival at Destination - Weekday (Mo = 1)'] = feature_vector_df['Arrival at Destination - Weekday (Mo = 1)'].astype('category')

    # Selecting columns to match training data
    feature_vector_df = feature_vector_df[['Platform Type', 'Personal or Business', 'Pickup - Day of Month',
                                            'Pickup - Weekday (Mo = 1)', 'Pickup - Time', 'Pickup Lat',
                                            'Pickup Long', 'Destination Lat',	'Destination Long', 'Distance (KM)',
                                             'Temperature', 'Precipitation in millimeters']]

    # Function to assign time values into time buckets
    def assign_time_category(delivery_time):
        dts = pd.DataFrame(['00:00:00', '6:00:00', '9:00:00', '12:00:00', '15:00:00', '18:00:00'])
        dts[0] = pd.to_datetime(dts[0]).dt.time
        if delivery_time >= dts[0][0] and delivery_time < dts[0][1]:
            return 'Early Morning'
        elif delivery_time >= dts[0][1] and delivery_time < dts[0][2]:
            return 'Morning'
        elif delivery_time >= dts[0][2]  and delivery_time < dts[0][3]:
            return 'Late Morning'
        elif delivery_time >= dts[0][3] and delivery_time < dts[0][4]:
            return 'Afternoon'
        elif delivery_time >= dts[0][4] and delivery_time < dts[0][5]:
            return 'Late Afternoon'
        else:
            return 'Evening'

    # Create new time bucket feature using the assign_time_category function
    feature_vector_df['Delivery Times'] = feature_vector_df['Pickup - Time'].apply(assign_time_category)
    feature_vector_df['Delivery Times'] = feature_vector_df['Delivery Times'].astype('category')

    # Drop 'Pickup - Time' because we have created 'Delivery Times' in its place
    feature_vector_df.drop('Pickup - Time', axis=1, inplace=True)

    # Fill missing precipitation values with 0
    feature_vector_df['Precipitation in millimeters'] = feature_vector_df['Precipitation in millimeters'].fillna(value=0)

    # Function to fill nulls with a column's mean value
    def mean(col):
      return col.fillna(col.mean())
    
    # Impute missing temperature based on delivery time
    feature_vector_df['Temperature'] = feature_vector_df.groupby(['Delivery Times'])['Temperature'].transform(mean)

    # Function to calculate a coordinate's distance from the CBD coordinate
    def distance_CBD(lat, long):
        return math.sqrt(((lat - -1.283526) ** 2) + ((long - 36.823269) ** 2))

    feature_vector_df['Distance_CBD_pickup'] = np.vectorize(distance_CBD)(feature_vector_df['Pickup Lat'],
                                    feature_vector_df['Pickup Long'])
    feature_vector_df['Distance_CBD_dest'] = np.vectorize(distance_CBD)(feature_vector_df['Destination Lat'],
                                    feature_vector_df['Destination Long'])
    
    # One-hot encoding for categorical data
    feature_vector_df = pd.get_dummies(feature_vector_df,
                                   columns=['Platform Type', 'Personal or Business', 'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)', 'Delivery Times'],
                                   prefix=['platformtype', 'personalbusiness', 'dayofmonth', 'weekday', 'pickuptimes'])
    
    # ------------------------------------------------------------------------
    
    predict_vector = feature_vector_df

    return predict_vector

def load_model(path_to_model:'assets/trained-models/random_forest_model.pkl'):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """

    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()

if __name__ == '__main__':
    print('inside main function')
    print('load in the data')
    data = pd.read_csv('Train.csv')
    print(data.head())
    predict_vector = _preprocess_data(data)
    print('data sent to preprocessing')
    print(predict_vector.head())
    y = data['Time from Pickup to Arrival']
    print('after preprocessing')
    print(data.head())
    #print('saving the preprocessing data')
    #predict_vector.to_csv(r'\Users\Monica\Documents\GitHub\regression-predict-api-template\processed_data.csv', index=False)
    print('testing data on random forest model')
    print('initialise random forest model')
    random_forest = RandomForestRegressor(n_estimators = 7)
    print('fitting the rfm model')
    X_train, X_test, y_train, y_test = train_test_split(predict_vector, y,test_size=0.2)
    random_forest.fit(X_train, y_train.values.ravel())
    y_pred_forest = random_forest.predict(predict_vector)
    print('model predicted successful')
    y_pred_forest = random_forest.predict(X_test)
    predictions = pd.DataFrame()
    predictions['Actual'] = y_test
    predictions['Predicted'] = y_pred_forest
    print(predictions.head())

    
