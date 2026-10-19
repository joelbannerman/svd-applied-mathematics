#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%Question 2

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import math


brain_img = Image.open("/Users/joelbannerman/Downloads/brain.jpg")
brain_grey = brain_img.convert("L") #converting to greyscale
brain_array = np.array(brain_grey.getdata(), float)

#reshape data (height, width)
brain_array = brain_array.reshape(brain_grey.size[0], brain_grey.size[1])

# Plotting each row as a separate line

# Visualizing as an image (heatmap-like)
plt.figure() # Create another new figure
plt.imshow(brain_array, cmap='grey', origin='upper')
plt.colorbar(label='Value')
plt.title("Original brain image")
plt.show()
#%%SVD
U, s, Vt = np.linalg.svd(brain_array, full_matrices=False)

def plot_spectrum(s, title, log=False):
    x = np.arange(1, len(s)+1)
    plt.figure()
    if log:
        plt.semilogy(x, s)
        plt.ylabel("Singular value σ_i (log scale)")
        plt.title(title + " (log)")
    else:
        plt.plot(x, s)
        plt.ylabel("Singular value σ_i")
        plt.title(title)
    plt.xlabel("k")
    plt.grid(True, which="both")
    plt.show()

plot_spectrum(s, "SVD Spectrum — Original Image", log=False)
plot_spectrum(s, "SVD Spectrum — Original Image", log=True)

# --------------- 3) Rotate image by +15° and compute SVD ---------------
def rotate_image_array(brain_array, degrees=15.0):
    # Use PIL to rotate with bilinear interpolation; expand=True keeps full content
    img = Image.fromarray(np.clip(brain_array, 0, 255).astype(np.uint8), mode="L")
    rotated = img.rotate(
        degrees,
        resample=Image.Resampling.BILINEAR,
        expand=True
    )
    return np.asarray(rotated, dtype=np.float64), rotated

brain_rotate, pil_rot = rotate_image_array(brain_array, degrees=15.0)

U_r, s_r, Vt_r = np.linalg.svd(brain_rotate)
plot_spectrum(s_r, "SVD Spectrum — Rotated (+15°) Image", log=False)
plot_spectrum(s_r, "SVD Spectrum — Rotated (+15°) Image", log=True)


rotated_brain= brain_grey.rotate(15)
rotated_brain_array = np.array(rotated_brain.getdata(), float)

rotated_brain_array = rotated_brain_array.reshape(rotated_brain.size[1], rotated_brain.size[0])


plt.figure() # Create another new figure
plt.imshow(rotated_brain_array, cmap='grey', origin='upper')
plt.colorbar(label='Value')
plt.title("Rotated Brain")
plt.show()

#%% correlation matrix 

XXT = brain_array @ brain_array.T
XTX = brain_array.T @ brain_array

plt.figure() # Create another new figure
plt.imshow(XXT, cmap='grey', origin='upper')
plt.colorbar(label='Value')
plt.title("Correlation Matrix XXT")
plt.show()

plt.figure() # Create another new figure
plt.imshow(XTX, cmap='grey', origin='upper')
plt.colorbar(label='Value')
plt.title("Correlation Matrix XTX")
plt.show()
#%%Question 3 Newton Raphson Method
A = np.array([[1,3,8],[2,4,7],[5,0,6]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f'The eigenvales for matrix A are {eigenvalues}')
print('the eigenvectors are: \n', eigenvectors)


#%%Question 4 Weather.csv Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


weather = pd.read_csv("Weather.csv")
max_min_mean = weather[['MaxTemp', 'MinTemp', 'MeanTemp']]

max_min_mean.plot(kind='scatter',
                  x='MinTemp',
                  y='MaxTemp',
                  figsize=(9,9),
                  color='black')

max_min_mean.plot(kind='scatter',
                  x='MinTemp',
                  y='MeanTemp',
                  figsize=(9,9),
                  color='red')

X = weather[['MinTemp']]
y = weather[['MaxTemp']]
y_mean = weather[['MeanTemp']]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)
X_mtrain, X_mtest, y_mtrain, y_mtest = train_test_split(X,y_mean,test_size=1/3,random_state=0)

regressor  = LinearRegression()
mregressor = LinearRegression()
mregressor.fit(X_mtrain, y_mtrain)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_mpred = mregressor.predict(X_mtest)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='red', label='Actual MaxTemp')
plt.plot(X_test, y_pred, color='blue', label='Predicted MaxTemp')
plt.title('Predicted vs Actual MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_mtest, y_mtest, color='green', label='Actual MeanTemp')
plt.plot(X_mtest, y_mpred, color='black', label='Predicted MeanTemp')
plt.title('Predicted vs Actual MeanTemp')
plt.xlabel('MinTemp')
plt.ylabel('MeanTemp')
plt.legend()

plt.tight_layout()
plt.show()

max_MSE = mean_squared_error(y_test, y_pred)
mean_MSE = mean_squared_error(y_mtest, y_mpred)


r_squared = r2_score(y_test,y_pred)
r_msquared = r2_score(y_mtest,y_mpred)
print(f'the R squared value for the MaxTemp model is {r_squared} and the MSE is {max_MSE}.')
print(f'the R squared value for the MeanTemp model is {r_msquared}. and the MSE is {mean_MSE}')

#%% Polynomial Regression Extension
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import math



X = weather[['MinTemp']]
y_max = weather[['MaxTemp']]
y_mean = weather[['MeanTemp']]

X_train, X_test, y_max_train, y_max_test = train_test_split(X, y_max, test_size=1/3, random_state=0)
X_mean_train, X_mean_test, y_mean_train, y_mean_test = train_test_split(X, y_mean, test_size=1/3, random_state=0)



degree = 7

poly = PolynomialFeatures(degree=degree)
X_poly_train = poly.fit_transform(X_train)
X_poly_test  = poly.transform(X_test)

poly_reg_max = LinearRegression().fit(X_poly_train, y_max_train)
poly_reg_mean = LinearRegression().fit(X_poly_train, y_mean_train)

y_max_pred = poly_reg_max.predict(X_poly_test)
y_mean_pred = poly_reg_mean.predict(X_poly_test)

# Evaluate
r2_max_poly  = r2_score(y_max_test, y_max_pred)
r2_mean_poly = r2_score(y_mean_test, y_mean_pred)

polymax_MSE = mean_squared_error(y_max_test, y_max_pred)
polymean_MSE = mean_squared_error(y_mean_test, y_mean_pred)
rmse_polymax = math.sqrt(polymax_MSE)
rmse_polymean = math.sqrt(polymean_MSE)
print(f"Polynomial Regression (degree={degree})")
print(f"MaxTemp:  R² = {r2_max_poly:.4f}")
print(f"MeanTemp: R² = {r2_mean_poly:.4f}")
print(f"Max RMSE = {rmse_polymax}")
print(f"Mean RMSE = {rmse_polymean}")


# Plot for MaxTemp
plt.figure(figsize=(10, 5))

sort_idx = X_test['MinTemp'].argsort()
X_sorted = X_test.iloc[sort_idx]
y_max_sorted = y_max_test.iloc[sort_idx]
y_max_pred_sorted = y_max_pred[sort_idx]

plt.subplot(1, 2, 1)
plt.scatter(X_test, y_max_test, color='red', label='Actual MaxTemp')
plt.plot(X_sorted, y_max_pred_sorted, color='blue', label=f'Polynomial (deg={degree}) Fit')
plt.title('Polynomial Regression: MaxTemp vs MinTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.legend()

# Plot for MeanTemp
y_mean_sorted = y_mean_test.iloc[sort_idx]
y_mean_pred_sorted = y_mean_pred[sort_idx]

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_mean_test, color='green', label='Actual MeanTemp')
plt.plot(X_sorted, y_mean_pred_sorted, color='black', label=f'Polynomial (deg={degree}) Fit')
plt.title('Polynomial Regression: MeanTemp vs MinTemp')
plt.xlabel('MinTemp')
plt.ylabel('MeanTemp')
plt.legend()

plt.tight_layout()
plt.show()
#%%Question 5 K-means clustering


from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 

#PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

b = StandardScaler().fit_transform(X) #standardize data with mean and standard deviation

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(b)
principalDF = pd.DataFrame(data = principalComponents,columns = ['principal component 1', 'principal component 2'])

print(principalDF)

finalDF = pd.concat([principalDF,y],axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['orange', 'g', 'orchid']
for target, color in zip(targets,colors):
    indicesToKeep = finalDF['class'] == target
    ax.scatter(finalDF.loc[indicesToKeep, 'principal component 1']
               , finalDF.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

#%%Kmeans cluster
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

plt.scatter(principalDF['principal component 1'], principalDF['principal component 2'])

km = KMeans(n_clusters=3)

y_predicted= km.fit_predict(principalDF[['principal component 1', 'principal component 2']])


principalDF['cluster'] = y_predicted
principalDF.head()


df1 = principalDF[principalDF.cluster == 0]
df2 = principalDF[principalDF.cluster == 1]
df3 = principalDF[principalDF.cluster == 2]




plt.scatter(df1['principal component 1'],df1['principal component 2'], color='green', label = 'iris-versicolor')
plt.scatter(df2['principal component 1'],df2['principal component 2'], color='orange', label = 'iris-setosa')
plt.scatter(df3['principal component 1'],df3['principal component 2'], color='purple', label = 'iris-virginica')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color = 'red', marker='*', label='centroid')
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.legend()
plt.title('k-means clustering')
#%%Elbow plot
k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(principalDF[['principal component 1', 'principal component 2']])
    sse.append(km.inertia_)
 
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.title('Elbow plot to determine K')
plt.plot(k_rng,sse)

































