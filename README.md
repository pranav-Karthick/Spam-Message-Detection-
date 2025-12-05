📧 Spam Classification Using Logistic Regression

A machine learning project that classifies SMS messages as spam or not spam using Logistic Regression, Sigmoid function, and Softmax probability distribution.
The model is trained on the spam.csv dataset and predicts spam status for 10 random messages selected from the dataset.

📝 Project Overview

This project demonstrates:

Text preprocessing using TF-IDF Vectorizer

Training a Logistic Regression classifier

Applying Sigmoid function for binary probability

Applying Softmax function for probability distribution

Predicting spam / not spam for 10 randomly selected messages

Displaying prediction probabilities clearly

This project is implemented in Python (Google Colab / Jupyter Notebook).

📂 Dataset

The project uses the standard spam.csv SMS spam collection dataset containing:

Column	Description
v1	Label (ham/spam)
v2	SMS message text

During preprocessing:

ham → not spam

spam → spam

Extra columns are removed

🔧 Tech Stack

Python

Pandas

NumPy

Scikit-Learn

Logistic Regression

Train–Test Split

TF-IDF Vectorizer

🚀 How It Works
🔹 1. Load and clean the dataset

Only required columns are kept and labels are converted.

🔹 2. Train–Test Split

80% for training, 20% for testing.

🔹 3. TF-IDF Vectorization

Converts text into numerical feature vectors.

🔹 4. Train Logistic Regression

Model learns to classify spam vs not spam.

🔹 5. Select 10 random samples

Random messages from the dataset are selected for testing.

🔹 6. Apply Sigmoid

Used to calculate the probability that a message is spam.

🔹 7. Apply Softmax

Converts scores into a probability distribution:

Probability of spam

Probability of not spam

🔹 8. Predict Labels

If sigmoid_prob > 0.5 → spam
Else → not spam

📌 Code Snippet (Main Prediction Part)
sample_df = df.sample(10, random_state=42).copy()
sample_tfidf = vectorizer.transform(sample_df['message'])

raw_scores = model.decision_function(sample_tfidf)
sigmoid_scores = sigmoid(raw_scores)
softmax_scores = np.array([softmax([1-s, s]) for s in sigmoid_scores])

sample_df['predicted_label'] = np.where(
    sigmoid_scores > 0.5, 'spam', 'not spam'
)

📊 Output

Final output contains:

message	sigmoid_prob_spam	softmax_spam	predicted_label
"Congratulations! You won…"	0.92	0.95	spam
"Ok I'll call later"	0.08	0.04	not spam
📁 Project Files
File	Description
spam.csv	Dataset file
notebook.ipynb	Full Google Colab/Jupyter code
README.md	Project documentation
🧠 Concepts Used
✔ Logistic Regression

Used for binary classification (spam/not spam)

✔ Sigmoid Function

Converts raw score into probability


σ(z)=1/1+e^−z

	​pip install numpy pandas scikit-learn
✔ Softmax

Probability distribution for two classes

✔ TF-IDF

Transforms text into machine-readable features

📦 Installation
pip install numpy pandas scikit-learn

▶️ Run the Project

Open the notebook in Google Colab:

Upload spam.csv

Run all cells

View predictions for 10 random messages

🏁 Conclusion

This project successfully classifies SMS messages using logistic regression and demonstrates how sigmoid and softmax functions can be used to compute probabilities for classification tasks.