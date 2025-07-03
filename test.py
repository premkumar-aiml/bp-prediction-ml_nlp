import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

patient_data = pd.read_csv('E:\\BITS_Masters_In_DataScience\\Dissertation\\Practice\\kaggle_Data\\framingham_heart_study_updated.csv')
# Print the patient_data to get more details
print(patient_data.shape)
for column in patient_data.columns:
    print(column)

# Totally we have 16 columns in the dataset.
# However, education and tenYearCHD columns are not useful for our analysis. Hence, we can drop those columns
dropped_patient_data = patient_data.drop(['education','TenYearCHD'],axis=1)
print(dropped_patient_data.head())

# Check null values in the dataset
print(dropped_patient_data.isnull().sum())

# There are few null values in the dataset: BpMeds = 53, totChol=50, BMI=19, heartRate=1, glucose=388
final_patientData = dropped_patient_data.dropna()
print(final_patientData.isnull().sum())

# Get rid of other columns whose values are strong giveaways that a person is likely to have high blood pressure.
final_patientData = final_patientData.drop(['BPMeds','prevalentHyp','sysBP'], axis=1)

# Performing exploratory analysis to understand the data better

# Find the relationship between Smoking vs diaBP using boxplot representation
plt.figure(figsize=(12,8))
sns.boxplot(x='currentSmoker', y='diaBP',  hue='currentSmoker', data=final_patientData, palette='Set2',  legend=False  )
plt.show()

#  Find the relationship between diabetes vs diaBP using barplot representation
f,ax = plt.subplots(figsize=(12,8))
sns.barplot(data=final_patientData, x='diabetes', y='diaBP', hue='diabetes', palette='Set2', legend=False)

for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 4, round(rect.get_height(),2),horizontalalignment='center', fontsize=12)
plt.show()

# ----------------
# Find the relationship between PrevalentStroke vs diaBP using barplot representation
f,ax = plt.subplots(figsize = (12,8))
sns.barplot(data=final_patientData, x='prevalentStroke', y='diaBP',hue = 'prevalentStroke',palette='Set2', legend=False)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 4, round(rect.get_height(),2),horizontalalignment='center', fontsize=12)
plt.show()

# Using Spearman's correlation coefficient compute the correlation of ordinal data
# Removing nominal fields cigsPerDay,male from the dataset

final_patientData_new = final_patientData.drop(['male','cigsPerDay'], axis =1)
patientData_corr = final_patientData_new.corr(method='spearman')
print(patientData_corr)

# Heatmap for the correlation matrix
f,ax = plt.subplots(figsize=(12,8))
sns.heatmap(patientData_corr, annot=True)
plt.show()

# convert floating point value to integers
final_patientData['cigsPerDay'] = final_patientData['cigsPerDay'].astype(int)
print(final_patientData.head())

final_patientData.to_csv('E:\\BITS_Masters_In_DataScience\\Dissertation\\Practice\\kaggle_Data\\processed_data_blood_pressure_updated.csv')