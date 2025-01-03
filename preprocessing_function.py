import pandas as pd

def preprocessing_function(test_data, original_data):
    # Take a large portion of the data as historical data
    historical_data = original_data.sample(frac=0.8, random_state=1).copy()  # Take 80% of the original data
    
    # Mark the test data and historical data
    test_data['is_test'] = 1
    historical_data['is_test'] = 0
    
    # Concatenate historical_data with test_data
    data = pd.concat([historical_data, test_data], ignore_index=True)  # Avoid index conflict

    # Convert types into categories and dates
    data[['TransactionType', 'Channel', 'CustomerOccupation']] = \
        data[['TransactionType', 'Channel', 'CustomerOccupation']].astype('category')
    data[['TransactionDate', 'PreviousTransactionDate']] = \
        data[['TransactionDate', 'PreviousTransactionDate']].apply(pd.to_datetime)

    # Extract features from dates
    data['TD_Year'] = data['TransactionDate'].dt.year
    data['TD_Month'] = data['TransactionDate'].dt.month
    data['TD_Day'] = data['TransactionDate'].dt.day
    data['TD_Hour'] = data['TransactionDate'].dt.hour
    data['TD_Weekday'] = data['TransactionDate'].dt.weekday

    data['PTD_Year'] = data['PreviousTransactionDate'].dt.year
    data['PTD_Month'] = data['PreviousTransactionDate'].dt.month
    data['PTD_Day'] = data['PreviousTransactionDate'].dt.day
    data['PTD_Hour'] = data['PreviousTransactionDate'].dt.hour
    data['PTD_Weekday'] = data['PreviousTransactionDate'].dt.weekday

    # Delete unused columns
    data.drop(columns=['TransactionDate', 'PreviousTransactionDate', 'PTD_Year', 'PTD_Month', 'PTD_Day', 'PTD_Hour', 'PTD_Weekday'], inplace=True)

    # Add features for activity metrics
    data['AccountActivity'] = data.groupby('AccountID')['TransactionID'].transform('count')
    data['MerchantActivity'] = data.groupby('MerchantID')['TransactionID'].transform('count')
    data['DeviceActivity'] = data.groupby('DeviceID')['TransactionID'].transform('count')
    data['IPActivity'] = data.groupby('IP Address')['TransactionID'].transform('count')
    data['LocationActivity'] = data.groupby('Location')['TransactionID'].transform('count')
    data['LoginActivity'] = data['LoginAttempts'].apply(lambda x: 0 if x == 1 else 1)

    # Delete unused columns
    data.drop(columns=['AccountID', 'MerchantID', 'DeviceID', 'IP Address', 'Location', 'LoginAttempts'], inplace=True)

    # Restore the test_data after transformations
    test_data_transformed = data.query('is_test == 1').copy()

    # Drop the temporary column after extraction
    test_data_transformed.drop(columns=['is_test'], inplace=True)

    # Set index
    test_data_transformed.set_index('TransactionID', inplace=True)

    return test_data_transformed
