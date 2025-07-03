import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# Load data
processed_patient_Data = pd.read_csv('E:\\BITS_Masters_In_DataScience\\Dissertation\\Practice\\kaggle_Data\\processed_data_blood_pressure.csv')

# Split features and target
X = processed_patient_Data.drop(['diaBP'], axis=1)
Y = processed_patient_Data['diaBP']

#  Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=33)

# Feature split
numerical_cols = ['age', 'cigsPerDay', 'totChol', 'BMI', 'heartRate', 'glucose']
categorical_cols = ['male', 'currentSmoker', 'prevalentStroke', 'diabetes']

X_train_numerical = X_train[numerical_cols]
X_test_numerical = X_test[numerical_cols]

X_train_categorical = X_train[categorical_cols].reset_index(drop=True)
X_test_categorical = X_test[categorical_cols].reset_index(drop=True)

# Standardize the numerical features
scaler = StandardScaler()

X_train_numerical = pd.DataFrame(scaler.fit_transform(X_train_numerical), columns=X_train_numerical.columns)
X_test_numerical  = pd.DataFrame(scaler.fit_transform(X_test_numerical), columns=X_test_numerical.columns)

# Reset the index on categorical and numerical data to concatenate them for training data
X_test_categorical.reset_index(drop=True, inplace=True)
X_train_numerical.reset_index(drop=True,inplace=True)
X_train = pd.concat([X_train_numerical, X_train_categorical], axis=1)

# For Testing data
X_test_categorical.reset_index(drop=True, inplace=True)
X_test_numerical.reset_index(drop=True, inplace=True)
X_test = pd.concat([X_test_numerical,X_test_categorical],axis=1)


# Building the models
def build_model(regressor, X_train, Y_train, X_test, Y_test):
 model = regressor.fit(X_train,Y_train)
 Y_pred = regressor.predict(X_test)

# Once the value is predicted compute the R square score of the model for training and test data
# R square score is a measure of how well the models fits the data.

 training_score  = model.score(X_train, Y_train)
 testing_Score = model.score(X_test, Y_test)
 result_dict = {'Training Score': training_score,
                'Testing Score' : testing_Score}
 return model, result_dict

def performance_metrics_report():
    for key in report_log:
        print('-'*50)
        print('Regression Model -', key)
        print('-' *50)

        # print()
        for score in report_log[key]:
            print(score, report_log[key][score])
report_log = dict()

def get_user_input():
    print("Please enter the following patient details to calculate the patient's Diastlic Pressure:")
    input_data = {}
    input_data['age'] = float(input("Age:  "))
    input_data['cigsPerDay'] = float(input("Cigarettes per Day: "))
    input_data['totChol'] = float(input("Total Cholesterol: "))
    input_data['BMI'] = float(input("BMI: "))
    input_data['heartRate'] = float(input("Heart Rate: "))
    input_data['glucose'] = float(input("Glucose Level: "))
    input_data['male'] = int(input("Gender (1 = Male, 0 = Female): "))
    input_data['currentSmoker'] = int(input("Current Smoker? (1 = Yes, 0 = No): "))
    input_data['prevalentStroke'] = int(input("Prevalent Stroke? (1 = Yes, 0 = No): "))
    input_data['diabetes'] = int(input("Diabetes? (1 = Yes, 0 = No): "))
    return pd.DataFrame([input_data])
def predict_diastolic_bp():
    new_df = get_user_input()
    age = new_df['age'].iloc[0]
    gender = "Male" if new_df['male'].iloc[0] == 1 else "Female"
    # Split into numerical and categorical parts
    new_numerical = new_df[numerical_cols]
    new_categorical = new_df[categorical_cols]
    # Standardize numerical features using the SAME scaler as test data
    new_numerical_scaled = pd.DataFrame(scaler.transform(new_numerical), columns=numerical_cols)
    # Combine both
    new_input = pd.concat([new_numerical_scaled, new_categorical], axis=1)
    # Predict using KNN and RF
    knn_prediction = knn_model.predict(new_input)
    rf_prediction = rf_model.predict(new_input)
    print(f"Calculated diastolic blood pressure {knn_prediction[0]}")
    return knn_prediction[0], age,gender

# Finally print the performance metrics report of the all te models used for predict the Diastolic BP
knn_model, knn_report = build_model(KNeighborsRegressor(), X_train, Y_train, X_test, Y_test)
report_log['KNeighborsRegressor'] = knn_report

rf_model, rf_report = build_model(RandomForestRegressor(), X_train, Y_train, X_test, Y_test)
report_log['RandomForestRegressor'] = rf_report

lr_model, lr_report = build_model(LinearRegression(), X_train, Y_train, X_test, Y_test)
report_log['LinearRegression'] = lr_report

performance_metrics_report()



