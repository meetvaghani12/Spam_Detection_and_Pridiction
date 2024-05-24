import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model

# Loading the data from csv file to a pandas DataFrame
raw_mail_data = pd.read_csv('/content/mail_data.csv')

# Replace the null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

# Label spam mail as 0 and ham mail as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separating the data as texts and labels
X = mail_data['Message']
Y = mail_data['Category']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Text vectorization
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train).toarray()  # Convert to dense array
X_test_features = feature_extraction.transform(X_test).toarray()  # Convert to dense array

# Convert Y_train and Y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Build the neural network model
model = Sequential()
model.add(Dense(64, input_shape=(X_train_features.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(X_train_features, Y_train, epochs=10, batch_size=32, validation_split=0.1)

# Save the trained model
model.save('spam_detection_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test_features, Y_test)
print("Test Accuracy:", accuracy)

# Prediction on test data
predictions = model.predict(X_test_features)
predictions_classes = (predictions > 0.5).astype('int').flatten()

# Text vectorization
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train).toarray()  # Convert to dense array
X_test_features = feature_extraction.transform(X_test).toarray()  # Convert to dense array

# Save the fitted vectorizer
import joblib
joblib.dump(feature_extraction, 'tfidf_vectorizer.pkl')

# Accuracy on test data
accuracy_on_test_data = accuracy_score(Y_test, predictions_classes)
print('Accuracy on test data:', accuracy_on_test_data)