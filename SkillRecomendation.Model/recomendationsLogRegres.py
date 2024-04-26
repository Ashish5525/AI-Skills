import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the SpaCy English language model
nlp = spacy.load("en_core_web_sm")

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('job_postings.csv')

# Load job data with relevant information (job description, skills, qualifications, salary range)
job_data = pd.read_csv('jobData.csv')

# Define the list of target job titles
target_job_titles = ['Software Developer', 'Business Manager', 'Professor', 'Doctor/Nurse', 'Stockbroker', 'Engineer']

# Filter DataFrame to include only target job titles
df_target = df[df['Job_Title'].isin(target_job_titles)]

# Prepare labeled dataset for classification
X = df_target['Job_Title']  # Features (job titles)
y = df_target['Category']   # Target labels (job categories)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature engineering (text vectorization using SpaCy) and train a classification model
X_train_vectorized = [doc.vector for doc in nlp.pipe(X_train)]
X_test_vectorized = [doc.vector for doc in nlp.pipe(X_test)]

# Train a logistic regression classifier
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Evaluate the model
accuracy = model.score(X_test_vectorized, y_test)
print("Model Accuracy:", accuracy)

# Use the trained model to predict job categories for new job titles
new_job_titles = ['Software Developer', 'Manager', 'Doctor', 'Engineer']
new_job_titles_vectorized = [doc.vector for doc in nlp.pipe(new_job_titles)]
predicted_categories = model.predict(new_job_titles_vectorized)

# Retrieve relevant information based on predicted categories
for title, category in zip(new_job_titles, predicted_categories):
    print(f"Job Title: {title}, Predicted Category: {category}")
    # Retrieve relevant information based on the predicted category
    relevant_info = job_data[job_data['Category'] == category]
    print(relevant_info.head())  # Print first few rows as an example