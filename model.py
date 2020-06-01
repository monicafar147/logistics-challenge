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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
from pprint import pprint
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Scaling
from sklearn.preprocessing import StandardScaler

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Machine learning models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingRegressor

# Saving model with pickle
import pickle

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
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    feature_vector_df.columns = [col.replace("-"," ") for col in feature_vector_df.columns]
    feature_vector_df.columns = [col.replace("(Mo = 1)"," ") for col in feature_vector_df.columns]
    feature_vector_df.columns = [col.replace(" ","") for col in feature_vector_df.columns]

    # Convert time columns to datetime 
    feature_vector_df['PickupTime'] = pd.to_datetime(feature_vector_df['PickupTime']).dt.time
    feature_vector_df['PlacementTime'] = pd.to_datetime(feature_vector_df['PlacementTime']).dt.time
    feature_vector_df['ConfirmationTime'] = pd.to_datetime(feature_vector_df['ConfirmationTime']).dt.time
    feature_vector_df['ArrivalatPickupTime'] = pd.to_datetime(feature_vector_df['ArrivalatPickupTime']).dt.time
    #feature_vector_df['ArrivalatDestinationTime'] = pd.to_datetime(feature_vector_df['ArrivalatDestinationTime']).dt.time

    #feature_vector_df = feature_vector_df[feature_vector_df['TimefromPickuptoArrival'] > 60]

    # Converting relevant columns into categorical data types
    feature_vector_df['VehicleType'] = feature_vector_df['VehicleType'].astype('category')
    feature_vector_df['PlatformType'] = feature_vector_df['PlatformType'].astype('category')
    feature_vector_df['PersonalorBusiness'] = feature_vector_df['PersonalorBusiness'].astype('category')

    feature_vector_df['PlacementDayofMonth'] = feature_vector_df['PlacementDayofMonth'].astype('category')
    feature_vector_df['PlacementWeekday'] = feature_vector_df['PlacementWeekday'].astype('category')
    feature_vector_df['ConfirmationDayofMonth'] = feature_vector_df['ConfirmationDayofMonth'].astype('category')
    feature_vector_df['ConfirmationWeekday'] = feature_vector_df['ConfirmationWeekday'].astype('category')

    feature_vector_df['ArrivalatPickupDayofMonth'] = feature_vector_df['ArrivalatPickupDayofMonth'].astype('category')
    feature_vector_df['ArrivalatPickupWeekday'] = feature_vector_df['ArrivalatPickupWeekday'].astype('category')
    #feature_vector_df['ArrivalatDestinationDayofMonth'] = feature_vector_df['ArrivalatDestinationDayofMonth'].astype('category')
    #feature_vector_df['ArrivalatDestinationWeekday'] = feature_vector_df['ArrivalatDestinationWeekday'].astype('category')

    #selecting training features
    training_features = feature_vector_df.iloc[:, :-1]
    training_features[['PickupDayofMonth', 'PickupWeekday']] = training_features[['PickupDayofMonth', 'PickupWeekday']].astype('category')
    training_features[['PlatformType', 'PersonalorBusiness']] = training_features[['PlatformType', 'PersonalorBusiness']].astype('category')
    
    # Selecting columns to match training data
    training_features = training_features[['PlatformType', 'PersonalorBusiness',
                                       'PickupDayofMonth', 'PickupWeekday',
                                       'PickupTime', 'PickupLat',	'PickupLong',
                                       'DestinationLat',	'DestinationLong',
                                       'Distance(KM)', 'Temperature',
                                       'Precipitationinmillimeters']]

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

    # Create new time bucket feature based on assign_time_category function
    training_features['DeliveryTimes'] = training_features['PickupTime'].apply(assign_time_category)
    training_features['DeliveryTimes'] = training_features['DeliveryTimes'].astype('category')

    # Dropping 'Pickup - Time' because we have created 'Delivery Times' in its place
    training_features.drop('PickupTime', axis=1, inplace=True)

    # Fill missing precipitation values with 0
    training_features['Precipitationinmillimeters'] = training_features['Precipitationinmillimeters'].fillna(value=0)
   
    # Impute missing temperature based on delivery time
    # Function to fill nulls with a column's mean value
    def mean(col):
      return col.fillna(col.mean())

    # Impute missing temperature based on delivery time
    training_features['Temperature'] = training_features.groupby(['DeliveryTimes'])['Temperature'].transform(mean)

    # Function to calculate a coordinate's distance from the CBD coordinate
    def distance_CBD(lat, long):
      return math.sqrt(((lat - -1.283526) ** 2) + ((long - 36.823269) ** 2))

    training_features['DistanceCBDpickup'] = np.vectorize(distance_CBD)(training_features['PickupLat'],training_features['PickupLong'])
    training_features['DistanceCBDdest'] = np.vectorize(distance_CBD)(training_features['DestinationLat'],training_features['DestinationLong'])
    
    # One-hot encoding for categorical data
    training_features = pd.get_dummies(training_features,
                                   columns=['PlatformType', 'PersonalorBusiness', 'PickupDayofMonth', 'PickupWeekday', 'DeliveryTimes'],
                                   prefix=['platformtype', 'personalbusiness', 'dayofmonth', 'weekday', 'pickuptimes'])

    model_columns = ['PickupLat', 'PickupLong', 'DestinationLat', 'DestinationLong',
       'Distance(KM)', 'Temperature', 'Precipitationinmillimeters',
       'DistanceCBDpickup', 'DistanceCBDdest', 'platformtype_1',
       'platformtype_2', 'platformtype_3', 'platformtype_4',
       'personalbusiness_Business', 'personalbusiness_Personal',
       'dayofmonth_1', 'dayofmonth_2', 'dayofmonth_3', 'dayofmonth_4',
       'dayofmonth_5', 'dayofmonth_6', 'dayofmonth_7', 'dayofmonth_8',
       'dayofmonth_9', 'dayofmonth_10', 'dayofmonth_11', 'dayofmonth_12',
       'dayofmonth_13', 'dayofmonth_14', 'dayofmonth_15', 'dayofmonth_16',
       'dayofmonth_17', 'dayofmonth_18', 'dayofmonth_19', 'dayofmonth_20',
       'dayofmonth_21', 'dayofmonth_22', 'dayofmonth_23', 'dayofmonth_24',
       'dayofmonth_25', 'dayofmonth_26', 'dayofmonth_27', 'dayofmonth_28',
       'dayofmonth_29', 'dayofmonth_30', 'dayofmonth_31', 'weekday_1',
       'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
       'weekday_7', 'pickuptimes_Afternoon', 'pickuptimes_Evening',
       'pickuptimes_Late Afternoon', 'pickuptimes_Late Morning',
       'pickuptimes_Morning']

    for col in model_columns:
          if col not in training_features.columns:
               training_features[col] = 0

    # ------------------------------------------------------------------------
    
    predict_vector = training_features

    return predict_vector
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
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
