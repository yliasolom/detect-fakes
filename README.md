### **Fake News Detection**


# This project aims to develop models that can distinguish between real and fake news articles based on their headlines. The project includes the implementation of two different approaches: Tf-Idf with Logistic Regression and LightGBM models as well as DistillBert model.

### Dataset:
The dataset used for training and evaluation consists of the following files:

- train.csv: This file contains the labeled training data. Each row contains a news headline and its corresponding label (0 for real news, 1 for fake news).
- test.csv: This file is provided for demonstration purposes and contains news headlines. The labels in this file are set to 0.

### Here're some of the project's best features:

*   Approach 1:
**Tf-Idf with Logistic Regression and LightGBM**

- The Tf-Idf-based models utilize the following preprocessing steps:

1.Lowercasing the text data.
2.Removing punctuation and extra spaces.
3.Removing stop words.
4.Lemmatizing the text using the WordNetLemmatizer from the NLTK library.
5.Vectorizing the corpus using the TfidfVectorizer.

- Two models are trained using this approach:

**Logistic Regression:** This  baseline model achieves the highest accuracy on the test set (83.98%) with relatively low training time (0.40 seconds) and prediction time (0.10 seconds).

**LightGBM with Unigram:** This model achieves an accuracy of 70.96% on the test set but has longer training time (2.18 seconds) and prediction time (0.05 seconds) compared to Logistic Regression.

**LightGBM with Bigram:** This model achieves the lowest accuracy on the test set (60.89%) with intermediate training time (0.99 seconds) and prediction time (0.05 seconds).

The models are evaluated using the F1 score metric, and the results are as follows:

Model               | F1 Score
--------------------|---------
Logistic Regression | 0.840
LightGBM Unigram    | 0.710
LightGBM Bigram     | 0.609


**The Logistic Regression model achieves the highest F1 score of *0.840***


*   Approach 2: DistillBert

In this approach, the DistillBert architecture was used for fake news detection. The following observations were made during the training process:

1.The model's performance consistently improved as the training progressed.
2.The F1 score and AUC-ROC metrics increased, indicating better classification performance and discrimination ability between real and fake news.
3.The training loss decreased over epochs, suggesting that the model learned from the training data.
4.The validation loss initially decreased and then fluctuated but remained relatively low, indicating good generalization to unseen data.
5.After 30 epochs, the model achieved an **F1 score 0.8087** and an **AUC-ROC score 0.8098**.

- These results suggest that the DistillBert model effectively learns the underlying patterns in the data and makes accurate predictions. 

---------------------------------------------------

### Instructions

First create an environment and install the required packages using pip:
`python -m venv myenv`
For macOS and Linux:
`source myenv/bin/activate`
Install the packages:
`pip install -r requirements.txt`



- 1. **Training and Prediction with DistillBert**

- Train the model:

`python train_model_distillbert.py --train-file <path-to-train-file>`

This code is designed to train a sequence classification model using the DistilBERT architecture. This code trains a DistilBERT-based sequence classification model. Then code tokenizes the text data, creates PyTorch datasets, and performs training and validation. The model's performance is evaluated using F1 score and AUC-ROC. The loss curves are plotted and saved, and the trained model is saved as well.
To get predictions (being in a scr folder), use your tab-delimited file and see results in the report folder. Make sure you have a trained DistilBERT model available in the "models/distillbert_weights" directory before running this script.

- Get prediction:
`python predict_model_distillbert.py --test-file <your_test_file.csv>`

- 2. **Training and Prediction with LinearRegression and LightGBM:**

- Train the model:

`python train_model_reg.py --train-file <path-to-train-file>`

This code trains and saves two models, Logistic Regression and LightGBM, for fake news detection using a given training dataset. It performs the following steps:
Performs text preprocessing, including lowercasing, removal of stopwords, lemmatization, and stemming.
Uses grid search to find the best hyperparameters for both models.
Trains the models on the training data.
Evaluates the models using F1 score and plots the ROC curves.
Saves the trained models as pickle files in the "weights/basic" directory

- Get prediction:

`python predict_model_reg.py <test_file> <lr_model_file> <lgbm_model_file> <lr_predictions_file> <lgbm_predictions_file>`

Replace the placeholders test_file, lr_model_file, lgbm_model_file, lr_predictions_file, and lgbm_predictions_file with the actual file paths you want to use.


`test_file:` Path to the test file containing the data on which you want to make predictions.
`lr_model_file:` Path to the logistic regression model file (.joblib) that you want to use for prediction.
`lgbm_model_file:` Path to the LightGBM model file (.joblib) that you want to use for prediction.
`lr_predictions_file:` Path to save the logistic regression predictions as a CSV file.
`lgbm_predictions_file:` Path to save the LightGBM predictions as a CSV file.


For example:
`python predict_model_reg.py data/test_data.csv models/lr_model.joblib models/lgbm_model.joblib predictions/lr_predictions.csv predictions/lgbm_predictions.csv`



### Conclusion
In this project, I developed models to classify news headlines as real or fake. The logistic regression model demonstrated the highest accuracy on the test dataset, while the LightGBM models achieved slightly lower performance, although it is not tends to be overfitted. 

Based on the F1 scores, the Logistic Regression model achieved the highest performance among the Tf-Idf-based models.

In the second approach, the DistillBert architecture was used for fake news detection. During the training process, the model's performance consistently improved as the training progressed. The F1 score and AUC-ROC metrics increased, indicating better classification performance and discrimination ability between real and fake news. The training loss decreased over epochs, suggesting that the model learned from the training data. The validation loss initially decreased and then fluctuated but remained relatively low, indicating good generalization to unseen data. After 30 epochs, the DistillBert model achieved an F1 score of 0.8087 and an AUC-ROC score of 0.8098.

These results indicate that the DistillBert model effectively learns the underlying patterns in the data and makes accurate predictions.

In conclusion, this project successfully developed models for classifying news headlines as real or fake. The logistic regression model demonstrated the highest accuracy on the test dataset, while the LightGBM models achieved slightly lower performance. The choice of model may depend on specific requirements, such as speed or interpretability. The DistillBert model also showed promising results, indicating its effectiveness in learning patterns and making accurate predictions.

The choice of model may depend on the specific requirements, such as speed or interpretability.
