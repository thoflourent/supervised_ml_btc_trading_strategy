import numpy as np

def pct_change_numpy(arr, periods=1):
    """
    Calculate the percentage change over a specified number of periods for a NumPy array.

    Parameters:
        arr (numpy.ndarray): Input array.
        periods (int): Number of periods to calculate the percentage change.

    Returns:
        numpy.ndarray: Array of percentage changes, with `periods` leading NaNs if periods > 1.
    """
    arr = np.asarray(arr)  # Ensure input is a NumPy array
    if periods < 1:
        raise ValueError("Periods must be at least 1.")
    
    # Calculate percentage change
    result = (arr[periods:] - arr[:-periods]) / arr[:-periods]
    
    # Prepend NaN values to align the output size with the input
    return np.concatenate((np.full(periods, np.nan), result))

