import numpy as np

# Compute and return the average image for each digits
def compute_avg_digits(X, image_width):
    sum_all = np.zeros(image_width * image_width)
    for x in X:
        sum_all += x
    sum_all = sum_all/len(X)
    return sum_all

# we substract all the input with the average digits
def normalize_with_avg(X, avg_digit):
    normalized = []
    for x in X:
        r = x - avg_digit
        normalized.append(r)
    return normalized
