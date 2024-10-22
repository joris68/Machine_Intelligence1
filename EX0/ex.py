"""
Ex1: The Gaussian Distribution
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def vizualize_gaussian_distribution(neg_range : int, pos_range : int):
     mu, sigma = 0, 1 
     ran = abs(neg_range - pos_range)
     y = np.random.normal(mu, sigma, 1000)
     plt.hist(y, bins=40)
     plt.title('Gaussian Distribution')
     plt.xlabel('x')
     plt.ylabel('Probability Density')
     plt.grid()
     plt.show()
     

#vizualize_gaussian_distribution(-8,8)


'''


'''


mu = np.array([0, 0])  # Mean vector
sigma = np.array([[1, 0.8], [0, 0.8]])  # Covariance matrix

def bivariate_gaussian(x1, x2, mu, sigma):
    """Calculate the PDF of a bivariate Gaussian distribution."""
    det_sigma = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)
    d = len(mu)
    
    norm_const = 1 / (2 * np.pi * np.sqrt(det_sigma))
    x = np.array([x1, x2])
    diff = x - mu
    
    exponent = -0.5 * np.dot(diff.T, np.dot(inv_sigma, diff))
    return norm_const * np.exp(exponent)


x1_range = np.linspace(-8, 8, 400) 
x2_range = np.linspace(-8, 8, 400)
x1, x2 = np.meshgrid(x1_range, x2_range)

# Calculate the PDF over the grid
pdf_values = np.zeros(x1.shape)
for i in range(x1.shape[0]):
    for j in range(x1.shape[1]):
        pdf_values[i, j] = bivariate_gaussian(x1[i, j], x2[i, j], mu, sigma)

# Create a filled contour plot
plt.figure(figsize=(10, 8))
contour = plt.contourf(x1, x2, pdf_values, levels=50, cmap='viridis')
plt.colorbar(contour, label='Probability Density Function (PDF)')
plt.title('Bivariate Gaussian Distribution PDF')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')
plt.grid()
plt.show()


