# Entity-Recognition-using-Spacy
Create a customized model to identity the user defined entitys from the text
This Spacy model is created for the user input where user wants to update his email id to some new email for a specific account name and we have to retrieve that account name and email from the text.

There are 2 files -

In data.py we are creating the dataset for entitys account name and email. Please stick to the format of the dataset. If it is wrong then predictions will be wrong.

In Entity Recognizer.py we are creating the model.

Run Entity Recognizer.py to create a model and do predictions. If you want to create your own model for your own usecase then update data.py file in same from format with your dataset.
