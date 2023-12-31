import numpy as np

# Function to calculate the absolute error
def ErrorAbs(stored_predictions, selected_data):
    dataOriginal = np.array(selected_data)
    dataPredic = np.array(stored_predictions)
    if dataPredic.ndim == 2:
        dataPredic = np.squeeze(dataPredic)
        
    error_Abs = np.subtract(dataOriginal, dataPredic)
    error_Abs = np.absolute(error_Abs)

    return error_Abs

# Function to calculate the relative error
def ErrorRel(dataOriginal, dataErrorAbs):
    if dataErrorAbs.ndim == 2:
        dataErrorAbs = np.squeeze(dataErrorAbs)
    return np.divide(dataErrorAbs, dataOriginal)
