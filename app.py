import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Sample data creation (for demonstration)
data = {
    'Fever': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No'],
    'Chills': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes'],
    'Cough': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'Sputum_color': ['Yellow', 'None', 'Green', 'Clear', 'Yellow', 'Green', 'None', 'Yellow', 'Clear', 'Green'],
    'Chest_pain': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Headache': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'Muscle_pain': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'Fatigue': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'Diagnosis': ['Bacterial', 'Viral', 'Bacterial', 'Viral', 'Bacterial', 'Bacterial', 'Viral', 'Bacterial', 'Viral', 'Bacterial']
}
df = pd.DataFrame(data)

# Initialize separate LabelEncoders for each column
encoders = {col: LabelEncoder() for col in df.columns}

# Apply the LabelEncoders to each column
for col, encoder in encoders.items():
    df[col] = encoder.fit_transform(df[col])

# Split dataset into features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Function to classify new data
def classify_pneumonia(symptoms):
    symptoms_df = pd.DataFrame([symptoms])
    for col, encoder in encoders.items():
        if col != 'Diagnosis':  # We don't need to transform the target column
            symptoms_df[col] = encoder.transform(symptoms_df[col])
    prediction = model.predict(symptoms_df)
    return encoders['Diagnosis'].inverse_transform(prediction)[0]

# Streamlit app
st.title("Pneumonia Classifier: Bacterial or Viral")

st.write("Please enter the symptoms:")

# Create user input fields
fever = st.selectbox('Fever', ['Yes', 'No'])
chills = st.selectbox('Chills', ['Yes', 'No'])
cough = st.selectbox('Cough', ['Yes', 'No'])
sputum_color = st.selectbox('Sputum Color', ['Yellow', 'Green', 'Clear', 'None'])
chest_pain = st.selectbox('Chest Pain', ['Yes', 'No'])
headache = st.selectbox('Headache', ['Yes', 'No'])
muscle_pain = st.selectbox('Muscle Pain', ['Yes', 'No'])
fatigue = st.selectbox('Fatigue', ['Yes', 'No'])

# Create a dictionary of inputs
symptoms = {
    'Fever': fever,
    'Chills': chills,
    'Cough': cough,
    'Sputum_color': sputum_color,
    'Chest_pain': chest_pain,
    'Headache': headache,
    'Muscle_pain': muscle_pain,
    'Fatigue': fatigue
}

# Button to classify
if st.button('Classify'):
    diagnosis = classify_pneumonia(symptoms)
    st.write(f"The predicted diagnosis is: **{diagnosis}**")
