# Deep Learning: Charity Funding Predictor

## Background

The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. With my knowledge of Machine Learning and Deep Learning, I used the features in the provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, I have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. In this dataset there is a number of columns that capture metadata about each organization, such as the following:

* **EIN** and **NAME** — Identification columns
* **APPLICATION_TYPE** — Alphabet Soup application type
* **AFFILIATION** — Affiliated sector of industry
* **CLASSIFICATION** — Government organization classification
* **USE_CASE** — Use case for funding
* **ORGANIZATION** — Organization type
* **STATUS** — Active status
* **INCOME_AMT** — Income classification
* **SPECIAL_CONSIDERATIONS** — Special consideration for application
* **ASK_AMT** — Funding amount requested
* **IS_SUCCESSFUL** — Was the money used effectively or not

### Step 1: Preprocessed the data

Using Pandas and the Scikit-Learn’s `StandardScaler()`, I preprocessed the dataset in order to compile, train, and evaluate the Deep Learning model later on.

These are the preprocessing steps I completed:

1. Read the charity_data.csv using Pandas DataFrame, and identified what variables are considered the target & the features for my model.
2. Dropped the `EIN` and `NAME` columns.
3. Determined the number of unique values for each column.
4. For those columns that have more than 10 unique values, I determined the number of data points for each unique value.
6. Used the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then checked if the binning was successful.
7. Used `pd.get_dummies()` to encode categorical variables.

### Step 2: Compiled, Trained, and Evaluated the Model

Using TensorFlow, I designed a Deep Learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. Determined the number of neurons and layers in my model. Then, I compiled, trained, and evaluated the binary classification model to calculate the model’s loss and accuracy.

The steps I took:

1. Created a Deep Learning model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
2. Created the first hidden layer and chose an appropriate activation function.
3. Added more hidden layers with an appropriate activation function.
4. Created an output layer with an appropriate activation function.
5. Checked the structure of the model.
6. Compiled and trained the model.
7. Created a callback that saves the model's weights every 5 epochs into [callbacks_NONoptimized](callbacks_NONoptimized) folder.
8. Evaluated the model using the test data to determined the loss and accuracy.
9. Saved and exported my results to an HDF5 file, and named it `AlphabetSoupCharity.h5`.

### Step 3: Optimized the Model

Using TensorFlow, I optimized my model in order to achieve a target predictive accuracy higher than 75%. Created a new Jupyter Notebook file and named it `AlphabetSoupCharity_Optimzation.ipynb` to have the optimized version as a separate file. Saved and exported the results to an HDF5 file, and name it `AlphabetSoupCharity_Optimization.h5`. Also, saved the optimized model's weights every 5 epochs into [callbacks_optimized](callbacks_optimized) folder.

To achieve the goal I did the following:

* Brought back `NAME` column and dropped 3 other columns (+ the `EIN` column I dropped first time).
* Decreased the number of bins in `NAME` and `APPLICATION_TYPE` columns.
* Added another hidden layer and more neurons all hidden layers, and added one dropout layer.
* Reduced the number of epochs to the training regimen. 

### Step 4: Report on the Deep Learning Model

Wrote a [report](Report.md) on the performance of the model I created for Alphabet Soup.
My report contains Overview, Results, and Summary.