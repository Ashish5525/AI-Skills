import pandas as pd
import spacy

# Load the SpaCy English language model
nlp = spacy.load("en_core_web_sm")

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('job_postings.csv')

# Explore the structure and content of the DataFrame
print(df.head())

# Check for missing values and handle them if necessary
print(df.isnull().sum())

# Clean the text data (e.g., remove special characters, punctuation, etc.)
# Example: Remove special characters and convert text to lowercase
df['Job_Description'] = df['Job_Description'].str.replace('[^\w\s]', '').str.lower()

# Tokenize the text data using SpaCy
# Example: Tokenize job descriptions
df['tokenized_job_description'] = df['Job_Description'].apply(lambda x: [token.text for token in nlp(x)])

# Clean the text data (e.g., remove special characters, punctuation, etc.)
# Example: Remove special characters and convert text to lowercase
df['skills'] = df['skills'].str.replace('[^\w\s]', '').str.lower()

# Tokenize the text data using SpaCy
# Example: Tokenize job descriptions
df['tokenized_skills'] = df['skills'].apply(lambda x: [token.text for token in nlp(x)])

# Clean the text data (e.g., remove special characters, punctuation, etc.)
# Example: Remove special characters and convert text to lowercase
df['Qualifications'] = df['Qualifications'].str.replace('[^\w\s]', '').str.lower()

# Tokenize the text data using SpaCy
# Example: Tokenize job descriptions
df['tokenized_Qualifications'] = df['Qualifications'].apply(lambda x: [token.text for token in nlp(x)])

# Clean the text data (e.g., remove special characters, punctuation, etc.)
# Example: Remove special characters and convert text to lowercase
df['Salary_Range'] = df['Salary_Range'].str.replace('[^\w\s]', '').str.lower()

# Tokenize the text data using SpaCy
# Example: Tokenize job descriptions
df['tokenized_Salary_Range'] = df['Salary_Range'].apply(lambda x: [token.text for token in nlp(x)])

# Clean and tokenize skills, salary range, qualifications similarly

# You can perform similar preprocessing steps for other columns such as qualifications, job type, etc.

# Save the preprocessed data to a new CSV file (optional)
df.to_csv('preprocessed_job_postings.csv', index=False)