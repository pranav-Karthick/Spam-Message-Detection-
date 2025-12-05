📧 SMS Spam Classification Using Logistic Regression
This project demonstrates how to build a binary text classification model that identifies SMS messages as spam or not spam.
The project uses Logistic Regression, TF-IDF vectorization, and the Sigmoid activation function to calculate the final probability of a message being spam.

The goal is to show how logistic regression works at a mathematical level by manually applying the sigmoid function to model outputs.
The model also predicts 10 random messages from the dataset and displays their spam probability and classification result.

📝 About the Project

This project focuses on text classification using machine learning.
Given an SMS message, the model predicts whether the message is:

spam
not spam

The project uses:
    Logistic Regression
    TF-IDF vectorization
    Sigmoid activation (manually applied)
    10 random messages prediction
    This project is ideal for students learning:
    NLP basics
    Logistic regression internals
    Activation functions
    Preprocessing text for ML
    Real-world classification problems

📂 Dataset Details

The dataset used is the Spam SMS Collection Dataset (spam.csv) that includes:

Column	Description
v1	Label: ham (not spam) or spam
v2	SMS message text
Cleaning Performed:

✔ Only required columns kept
✔ Columns renamed to label and message
✔ Labels mapped → ham = 0, spam = 1
✔ Empty or irrelevant columns removed

🛠 Technologies Used

Python
Pandas
NumPy
Scikit-Learn
Logistic Regression
TF-IDF Vectorizer

🔄 Project Workflow

Load dataset
Preprocess text
Convert text to TF-IDF features
Split dataset: 80% train, 20% test
Train Logistic Regression model
Manually apply sigmoid function
Predict spam/not spam for 10 random messages
Display probabilities + final label

📘 Understanding the Sigmoid Function
Logistic Regression produces a raw output value called logit or decision score.

The sigmoid function converts this raw score into a probability:

[

\sigma(z) = \frac{1}{1 + e^{-z}}
]

Where:

z = model output

σ(z) = probability that the message is spam

Classification Decision:
Sigmoid  Output	Prediction
> 0.5	 spam
≤ 0.5	 not spam
🤖 Model Training Process
🔹 TF-IDF Vectorization

Converts text into numerical vectors based on word importance.

🔹 Logistic Regression

Learns the relationship between message features and their labels.

🔹 Sigmoid Activation

Manually applied to show how logistic regression actually converts output to probability.

🔟 10 Random Message Prediction

The model selects 10 random messages from the dataset and predicts:
Spam probability (sigmoid output)
Final classification (spam/not spam)
This demonstrates real-world prediction behavior.

📊 Project Output

The final output will look like:

message	sigmoid_prob_spam	predicted_label
"Congratulations! You won…"	0.928	spam
"Ok I will call later"	0.071	not spam
...	...	...

📜 Code Overview
The core part of the prediction:

raw_scores = model.decision_function(sample_tfidf)
sigmoid_scores = sigmoid(raw_scores)

sample_df['predicted_label'] = np.where(
    sigmoid_scores > 0.5,
    'spam',
    'not spam'
)

🚀 Future Enhancements

Add Softmax-based probability distribution

Add confusion matrix and accuracy score

Deploy as a web app using Streamlit or Gradio

Add stemming/lemmatization

Use more advanced models (SVM, Random Forest, Naive Bayes, LSTM)