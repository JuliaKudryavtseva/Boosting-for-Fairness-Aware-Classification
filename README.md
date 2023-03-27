# Boosting-for-Fairness-Aware-Classification
The project focuses on fairness-aware classification in machine learning, aiming to mitigate algorithms' bias towards minorities in various domains such as job hiring or loan credit. The problem is reduced to imbalanced classification and minority class prediction. The project involves reproducing, improving, and creating new algorithms and models for fairness-aware classification. The provided papers explore different algorithms for imbalanced classification using real-world datasets, including UCI Adult Census and UCI KDD Census. We will reproduce the models described in the articles and use the datasets for training and testing. The project aims to contribute to the current research on fairness-aware classification, which is a hot topic in the international machine learning community. By developing more accurate and fair algorithms, the project can lead to more ethical and unbiased decision-making processes in various domains.

# Prerequisites
We provide a file with all the required packages to reproduce the project on a local computer, the code must be executed strictly using sklearn==1.0.2, current versions such as 1.2.2 raise errors.
```
pip install -r requirements.txt
```
# Run experiments

It is possible to run the code from Colab and reproduce the results using the notebook: run reproduce_results.ipynb, it will automatically clone the repository into a temporary folder in Colab and install the prerequisites.

To run it locally, please attach to the requirements file and delete the first cell of the notebook above.

It is also possible to only get the DataFrame containing all our results by downloading the csv files from our folder "results" in this repsoitory.

# Datasets and Results

SMOTEBoost, AdaFair, CUSBoost, and RAMOBoost effectively eliminate bias in
four real-world datasets: Adult census, COMPAS, Bank, and KDD. 

These datasets are notorious for being imbalanced and biased against certain protected groups, and therefore pose significant challenges for building fair and accurate classifiers.

![image](https://user-images.githubusercontent.com/67862423/227771325-b6f3156a-add8-4c70-927d-3ce604cee8f3.png)

![image](https://user-images.githubusercontent.com/67862423/227772295-afb8fb73-b8ee-4625-b9ff-829426c19ce3.png)
