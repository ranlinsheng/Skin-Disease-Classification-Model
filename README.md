# Skin-Disease-Classification-Model

## Introduction
The existing skin disease classification models are trained with a dataset that has limited instances of dark skin tone patients. Consequently, the model developed is potentially biased and leads to inaccurate diagnosis for skin diseases especially patients with dark skin tone. Furthermore, skin disease manifests itself differently across different skin tones. Moreover, skin diseases appear visually similar thus are difficult to classify. Hence, there is a need to develop fair skin disease classification models for all people with different skin tones.

## DataSet
The Diverse Dermatology Image (DDI) dataset consists of skin lesion images diagnosed in Sandford Clinics from 2010 to 2020. The skin tones are determined using chart review of the in-person visit and consensus review by two board-certified dermatologists, while the skin diseases are biopsy proven diagnoses (Daneshjou Roxana et al., 2022). The total images in the dataset is 656. 
- [Data Set](https://drive.google.com/drive/folders/1lrumdZMu-Evdos4TBeiJqQgQXHFxTa9U?usp=sharing)

## Code
-  step1: loading data
```python
#data
    ddi_data_csv = pandas.read_csv('ddi_metadata.csv',index_col=0)
    ddi_data_csv.head()
```
-  step2: Image Preprocessing
```python
# Set the desired size
size = (500, 500)

# Iterate through all images in a directory
for file in os.listdir(path):
    if file.endswith(".jpg") or file.endswith(".png"):
        # Open the image
        im = Image.open(os.path.join(path, file))
        
        # Resize the image
        im_resized = im.resize(size)
        
         # Apply the median filter
        im_filtered = im.filter(ImageFilter.MedianFilter(size=5))
        
         # Convert the image to grayscale
        im_gray = im_filtered.convert("L")
        
         # Convert the image to a NumPy array
        im_array = np.array(im_gray)
        
        # Normalize the pixel values to the range 0-1
        im_array_normalized = im_array / 255.0
        
        # Convert the normalized array back to an image
        im_normalized = Image.fromarray(np.uint8(im_array_normalized * 255))
        
        # Save the normalized image
        im_normalized.save(os.path.join(path_after, file))
```


-  step3: Feature extraction
```python
def build_filters():
     filters = []
     ksize = [7,9,11,13,15,17] # gabor scale, 6
     lamda = np.pi/2.0         # Wave length
     for theta in np.arange(0, np.pi, np.pi / 4): #gabor direction, 0°, 45°, 90°, 135°, four in total
         for K in range(6):
            
                 kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
                 kern /= 1.5*kern.sum()
                 filters.append(kern)
     plt.figure(1)

     return filters
```
#Gabor Feature extraction
```python
#Gabor Feature extraction
def getGabor(img,filters):
    res = [] #Filtering results
    for i in range(len(filters)):
        # res1 = process(img, filters[i])
        accum = np.zeros_like(img)
        for kern in filters[i]:
            fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
            accum = np.maximum(accum, fimg, accum)
        res.append(np.asarray(accum)
    #For drawing filter effects
    plt.figure(2)
    for temp in range(len(res)):
        plt.subplot(4,6,temp+1)
        plt.imshow(res[temp], cmap='gray' )
   # plt.show()
    return res  #Returns the filtered results, which are 24 plots, arranged by gabor angle
filters = build_filters()
```
- step4: Classification
#### Random Forest
RF - Binary Model
```python
#cross-validation method
cv = KFold(n_splits=5, random_state=20, shuffle=True)

#build RF model
model_rfc = rfc()

#use k-fold CV to evaluate model
rf_accu = cross_val_score(model_rfc, data_x, data_y, scoring='accuracy', cv=cv, n_jobs=-1)
rf_pres = cross_val_score(model_rfc, data_x, data_y, scoring='precision', cv=cv, n_jobs=-1)
rf_f1 = cross_val_score(model_rfc, data_x, data_y, scoring='f1', cv=cv, n_jobs=-1)
rf_recal = cross_val_score(model_rfc, data_x, data_y, scoring='recall', cv=cv, n_jobs=-1)
rf_roc = cross_val_score(model_rfc, data_x, data_y, scoring='roc_auc', cv=cv, n_jobs=-1)
```

RF Multi-Class Model
```python
#cross-validation method
cv = KFold(n_splits=5, random_state=20, shuffle=True)

#build RF model
model_rfc = rfc()

#use k-fold CV to evaluate model
mrf_accu = cross_val_score(model_rfc, mul_data_x, mul_data_y, scoring='accuracy', cv=cv)
mrf_pres = cross_val_score(model_rfc, mul_data_x, mul_data_y, scoring='precision_macro', cv=cv)
mrf_f1 = cross_val_score(model_rfc, mul_data_x, mul_data_y, scoring='f1_macro', cv=cv)
mrf_recal = cross_val_score(model_rfc, mul_data_x, mul_data_y, scoring='recall_macro', cv=cv)
```

#### Decision Tree
DT - Binary Model
```python
#cross-validation method
cv = KFold(n_splits=5, random_state=20, shuffle=True)

#build RF model
model_dt = DecisionTreeClassifier()

#use k-fold CV to evaluate model
dt_accu = cross_val_score(model_dt, data_x, data_y, scoring='accuracy', cv=cv, n_jobs=-1)
dt_pres = cross_val_score(model_dt, data_x, data_y, scoring='precision', cv=cv, n_jobs=-1)
dt_f1 = cross_val_score(model_dt, data_x, data_y, scoring='f1', cv=cv, n_jobs=-1)
dt_recal = cross_val_score(model_dt, data_x, data_y, scoring='recall', cv=cv, n_jobs=-1)
dt_roc = cross_val_score(model_dt, data_x, data_y, scoring='roc_auc', cv=cv, n_jobs=-1)
```
DT - Multi Class Model
```python
#cross-validation method
cv = KFold(n_splits=5, random_state=20, shuffle=True)

#build DT model
model_dt = DecisionTreeClassifier()

#use k-fold CV to evaluate model
mdt_accu = cross_val_score(model_dt, mul_data_x, mul_data_y, scoring='accuracy', cv=cv)
mdt_pres = cross_val_score(model_dt, mul_data_x, mul_data_y, scoring='precision_macro', cv=cv)
mdt_f1 = cross_val_score(model_dt, mul_data_x, mul_data_y, scoring='f1_macro', cv=cv)
mdt_recal = cross_val_score(model_dt, mul_data_x, mul_data_y, scoring='recall_macro', cv=cv)
```

#### Support Vector Machine
SVM - Binary Model
```pythonfrom sklearn.svm import SVC

cv = KFold(n_splits=5, random_state=20, shuffle=True)

model_svm = SVC(kernel ='rbf', random_state = 0)

svm_accu = cross_val_score(model_svm, data_x, data_y, scoring='accuracy', cv=cv, n_jobs=-1)
svm_pres = cross_val_score(model_svm, data_x, data_y, scoring='precision', cv=cv, n_jobs=-1)
svm_f1 = cross_val_score(model_svm, data_x, data_y, scoring='f1', cv=cv, n_jobs=-1)
svm_recal = cross_val_score(model_svm, data_x, data_y, scoring='recall', cv=cv, n_jobs=-1)
svm_roc = cross_val_score(model_svm, data_x, data_y, scoring='roc_auc', cv=cv, n_jobs=-1)
```
SVM - Multi Class
```python
cv = KFold(n_splits=5, random_state=20, shuffle=True)

model_svm = SVC(kernel ='rbf', random_state = 0)

msvm_accu = cross_val_score(model_svm, mul_data_x, mul_data_y, scoring='accuracy', cv=cv)
msvm_pres = cross_val_score(model_svm, mul_data_x, mul_data_y, scoring='precision_macro', cv=cv)
msvm_f1 = cross_val_score(model_svm, mul_data_x, mul_data_y, scoring='f1_macro', cv=cv)
msvm_recal = cross_val_score(model_svm, mul_data_x, mul_data_y, scoring='recall_macro', cv=cv)
```

#### Naive Bayes
Binary - Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB

cv = KFold(n_splits=5, random_state=20, shuffle=True)

model_gNB = GaussianNB()

nb_accu = cross_val_score(model_gNB, data_x, data_y, scoring='accuracy', cv=cv, n_jobs=-1)
nb_pres = cross_val_score(model_gNB, data_x, data_y, scoring='precision', cv=cv, n_jobs=-1)
nb_f1 = cross_val_score(model_gNB, data_x, data_y, scoring='f1', cv=cv, n_jobs=-1)
nb_recal = cross_val_score(model_gNB, data_x, data_y, scoring='recall', cv=cv, n_jobs=-1)
nb_roc = cross_val_score(model_gNB, data_x, data_y, scoring='roc_auc', cv=cv, n_jobs=-1)
```
Multi Class - Naive Bayes
```python
cv = KFold(n_splits=5, random_state=20, shuffle=True)

model_gNB = GaussianNB()

mgNB_accu = cross_val_score(model_gNB, mul_data_x, mul_data_y, scoring='accuracy', cv=cv)
mgNB_pres = cross_val_score(model_gNB, mul_data_x, mul_data_y, scoring='precision_macro', cv=cv)
mgNB_f1 = cross_val_score(model_gNB, mul_data_x, mul_data_y, scoring='f1_macro', cv=cv)
mgNB_recal = cross_val_score(model_gNB, mul_data_x, mul_data_y, scoring='recall_macro', cv=cv)
```
```python

#### K-Nearest Neighbour
Binary - KNN
```python
from sklearn.neighbors import KNeighborsClassifier

cv = KFold(n_splits=5, random_state=20, shuffle=True)

model_kNN = KNeighborsClassifier(n_neighbors=3)

knn_accu = cross_val_score(model_kNN, data_x, data_y, scoring='accuracy', cv=cv, n_jobs=-1)
knn_pres = cross_val_score(model_kNN, data_x, data_y, scoring='precision', cv=cv, n_jobs=-1)
knn_f1 = cross_val_score(model_kNN, data_x, data_y, scoring='f1', cv=cv, n_jobs=-1)
knn_recal = cross_val_score(model_kNN, data_x, data_y, scoring='recall', cv=cv, n_jobs=-1)
knn_roc = cross_val_score(model_kNN, data_x, data_y, scoring='roc_auc', cv=cv, n_jobs=-1)
```
Multi Class - KNN
```python
model_kNN = KNeighborsClassifier(n_neighbors=3)

mknn_accu = cross_val_score(model_kNN, mul_data_x, mul_data_y, scoring='accuracy', cv=cv)
mknn_pres = cross_val_score(model_kNN, mul_data_x, mul_data_y, scoring='precision_macro', cv=cv)
mknn_f1 = cross_val_score(model_kNN, mul_data_x, mul_data_y, scoring='f1_macro', cv=cv)
mknn_recal = cross_val_score(model_kNN, mul_data_x, mul_data_y, scoring='recall_macro', cv=cv)
```

#### Extreme Gradient Boosting 
Binary - XGBoost
```python
from xgboost import XGBClassifier
model_XGB = XGBClassifier(n_estimators=110,max_depth=300,min_child_weight=1,verbosity =0,n_jobs=16)

XGB_accu = cross_val_score(model_XGB, data_x, data_y, scoring='accuracy', cv=cv, n_jobs=-1)
XGB_pres = cross_val_score(model_XGB, data_x, data_y, scoring='precision', cv=cv, n_jobs=-1)
XGB_f1 = cross_val_score(model_XGB, data_x, data_y, scoring='f1', cv=cv, n_jobs=-1)
XGB_recal = cross_val_score(model_XGB, data_x, data_y, scoring='recall', cv=cv, n_jobs=-1)
XGB_roc = cross_val_score(model_XGB, data_x, data_y, scoring='roc_auc', cv=cv, n_jobs=-1)
```

Multi Class - XGBoost
```python
model_XGB = XGBClassifier(n_estimators=110,max_depth=300,min_child_weight=1,verbosity =0,n_jobs=16)

mxgb_accu = cross_val_score(model_XGB, mul_data_x, mul_data_y, scoring='accuracy', cv=cv)
mxgb_pres = cross_val_score(model_XGB, mul_data_x, mul_data_y, scoring='precision_macro', cv=cv)
mxgb_f1 = cross_val_score(model_XGB, mul_data_x, mul_data_y, scoring='f1_macro', cv=cv)
mxgb_recal = cross_val_score(model_XGB, mul_data_x, mul_data_y, scoring='recall_macro', cv=cv)
```

## Results

| Classifiers | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| --- | --- | --- | --- | --- | --- |
| RF | 74.23 | 0.58 | 0.04 | 0.14 | 0.58 |
| DT | 61.11 | 0.28 | 0.32 | 0.30 | 0.52 |
| SVM | 74.23 | 0.40 | 0.01 | 0.02 | 0.55 |
| NB | 62.03 | 0.30 | 0.34 | 0.32 | 0.57 |
| KNN | 72.71 | 0.30 | 0.06 | 0.10 | 0.53 |
| XGBoost | 73.01 | 0.40 | 0.09 | 0.15 | 0.54 |

## Conclusion
The most critical step in medical health care is the proper diagnosis of disease therefore it is essential to build high accuracy models for skin disease classification. Moreover, skin algorithms developed must be capable of diagnosing skin disease on different skin tones. Therefore, using the suitable methodology for image pre-processing, feature extraction and using the right machine learning models is needed to build accurate skin disease classifiers. Although this study did not manage to develop high accuracy models, this study contributes by raising awareness in building fair skin disease classification models.

