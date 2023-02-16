# Milk-Grading-System-using-Machine-Learning

Milk Grading System Using Machine Learning
Project Description:
The purpose of grading milk is to separate the available supply of portable milk into classes differing in superiority. Nearly all food products are graded in some way. so that the consumer may select milk for particular purposes according to his desires and pocketbook. Certified milk is practically the only stable grade, rules for production being laid down by the American Association of Medical Milk Commissions. There is a serious question as to whether or not it is possible in the present state of the industry to enforce uniform grades universally. The main problem here is not just the feature sets and target sets but also the approach that is taken in solving these types of problems.
We will be using classification algorithms such as Decision tree, Random forest, svm, and Extra tree classifier. We will train and test the data with these algorithms. From this the best model is selected and saved in pkl format.
Technical Architecture:

 ![p1](https://user-images.githubusercontent.com/111130870/219393039-081bac51-ff5f-47fe-ac40-c39dc49751f3.jpg)
 
Project Objectives
By the end of this project you will:
•	Know fundamental concepts and techniques used for machine learning.
•	Gain a broad understanding about data.
•	Have knowledge on pre-processing the data/transformation techniques on outlier and some visualization concepts.

Project Flow

•	User interacts with the UI to enter the input.

•	Entered input is analysed by the model which is integrated.

•	Once model analyses the input the prediction is showcased on the UI
 
To accomplish this, we have to complete all the activities listed below,

•	Data collection

o	Collect the dataset or create the dataset

•	Visualising and analysing data

o	Univariate analysis

o	Bivariate analysis

o	Multivariate analysis

o	Descriptive analysis

•	Data pre-processing

o	Checking for null values

o	Handling outlier

o	Handling categorical data

o	Splitting data into train and test

•	Model building

o	Import the model building libraries

o	Initialising the model

o	Training and testing the model

o	Evaluating performance of model

o	Save the model

•	Application Building

o	Create an HTML file

o	Build python code

o	Run the application

Project Structure

Create the Project folder which contains files as shown below

![p2](https://user-images.githubusercontent.com/111130870/219394640-e67cfde2-f354-4656-97eb-22a22fa3d47d.png)


•	We are building a flask application which needs HTML pages stored in the templates folder and a python script app.py for scripting.

•	Model.pkl is our saved model. 

Further we will use this model for flask integration.
Pre Requisites

To complete this project, you must required following software’s, concepts and packages
Anaconda Navigator
 
Python Packages

Open anaconda prompt as administrator and install the required libraries by following the below instructions.

•	Type “pip install numpy” and click enter.

•	Type “pip install pandas” and click enter.

•	Type “pip install scikit-learn” and click enter.

•	Type ”pip install matplotlib” and click enter.

•	Type ”pip install scipy” and click enter.

•	Type ”pip install pickle-mixin” and click enter.

•	Type ”pip install seaborn” and click enter.

•	Type “pip install Flask” and click enter.

Prior Knowledge

You must have prior knowledge of following topics to complete this project

	ML Concepts
•	Supervisedlearning
•	Unsupervisedlearning
•	Regression and classification
•	Decisiontree
•	Randomforest
•	KNN
•	Svm
•	Extra tree classifier 
•	Evaluationmetrics


	Falsk Basics


Data Collection
•	ML depends heavily on data. It is the most crucial aspect that makes algorithm training possible. So this section allows you to download the required dataset.

•	There are many popular open sources for collecting the data. Eg: kaggle.com, UCI repository, etc.

Dataset

In this project we have used Milk Grading (1).csv data. This data is downloaded from kaggle.com. Please refer to the link given below to download the dataset

https://www.kaggle.com/datasets/prudhvignv/milk-grading

Data Preprocessing /Analysis

In this milestone, you need to complete all the below activities to build the model


	Importing the libraries

Import the necessary libraries as shown in the image. 

![p3](https://user-images.githubusercontent.com/111130870/219394829-bc53864a-7978-468b-b5b4-5cd4cc3d30d6.png)

	Read the Dataset

Our dataset format might be in .csv, excel files, .txt, .json, etc. We can read the dataset with the help of pandas.
In pandas we have a function called read_csv() to read the dataset. As a parameter we have to give the directory of the csv file.

![p4](https://user-images.githubusercontent.com/111130870/219394963-d25daed7-ecab-4026-a4a0-231b5d3cf178.png)


	Checking file size

![p5](https://user-images.githubusercontent.com/111130870/219395061-37cda2f6-2d4f-41dd-9277-c0050f3d4025.png)

 
	Data info
 
 ![p6](https://user-images.githubusercontent.com/111130870/219395134-b0ea2209-cf42-43b1-bd18-cff3cf13d2d5.png)

by  use the  info method to knowing the information about the dataset.
Visualization
•	Univariate analysis
•	Bivariate analysis
•	Multivariate analysis
•	Descriptive analysis
Univariate Analysis

In simple words, univariate analysis is understanding the data with a single feature. Here I have displayed the graph such as distplot .

The Seaborn package provides a wonderful function distplot. With the help of distplot, we can find the distribution of the feature.

![p7](https://user-images.githubusercontent.com/111130870/219395218-56600589-76e7-4cfc-85e4-24591e89a405.png)

 
Bivariate Analysis

To find the relation between two features we use bivariate analysis. Here we are visualising the relationship between Grade and Temperature.

Countplot is used here. As a 1st parameter we are passing x value and as a 2nd parameter we are passing hue value.

 ![p8](https://user-images.githubusercontent.com/111130870/219395467-f8cdb6c3-08af-44f9-ad94-1fdd8d1d52ca.png)

 Here we are visualising the relationship between Grade and pH.
 
Barplot is used here. As a 1st parameter we are passing x(Grade) value and as a 2nd parameter we are passing y(pH) value.

![p9](https://user-images.githubusercontent.com/111130870/219395417-6f6d3c25-5fa8-491f-8dc4-be1ed155a4f0.png)

 Here we are visualising the relationship between Grade and Colour.
 
Boxplot is used here. As a 1st parameter we are passing x(Grade) value and as a 2nd parameter we are passing y( Colour)value.

![p10](https://user-images.githubusercontent.com/111130870/219395578-a3056eaf-cc4c-444e-9c85-8dfdd11780d5.png)


In the sameway here we are visualising the relationship between Grade and Temperature.

Barplot is used here. As a 1st parameter we are passing x(Temperature) value and as a 2nd parameter we are passing y( Grade)value.

Multivariate Analysis

In simple words, multivariate analysis is to find the relation between multiple features. Here we have used swarmplot from the seaborn package.

 ![p12](https://user-images.githubusercontent.com/111130870/219395640-462bbc45-76f3-4794-a7b0-780319fc678e.png)

Here we are visualising the relationship between Temperature ,Grade and Taste.

Swarmplot is used here. As a 1st parameter we are passing x(Temperature) value and as a 2nd parameter we are passing y( Grade)value and huge(Taste) values.

Descriptive Analysis

Descriptive analysis is to study the basic features of data with the statistical process. Here pandas has a worthy function called describe. With this describe function we can understand the unique, top and frequent values of categorical features. And we can find mean, std, min, max and percentile values of continuous features.

![p13](https://user-images.githubusercontent.com/111130870/219395676-ebf40b57-0196-4023-99fa-302c2e0de8f2.png)


Here using the data.value_counts() to know whether the data set is overfitted or under fitted. In my data it is overfitted.
 
Using data.loc method to convert the continuous values to categorical values.


Outlier Detection

Checking for null values

Let’s find the shape of our dataset first. To find the shape of our data, the df.shape method is used. To find the data type, df.info() function is used.

 ![p16](https://user-images.githubusercontent.com/111130870/219395809-d6626a81-d7be-4eea-8233-f9afe9d06dc3.png)

For checking the null values, data.isnull() function is used. To sum those null values we use the .sum() function to it. From the below image we found that there are no null values present in our dataset.So we can skip handling of missing values step.


 ![p17](https://user-images.githubusercontent.com/111130870/219395845-de5dd5f6-b7ad-4f7d-af22-1ad40cbbae7a.png)


Let’s look for any outliers in the dataset
Activity 2: Handling outliers
With the help of boxplot, outliers are visualised. And here we are going to find the upper bound and lower bound of the pH feature with some mathematical formula.
From the below diagram, we could visualise that pH feature has outliers. Boxplot from seaborn library is used here.

![p18](https://user-images.githubusercontent.com/111130870/219395883-f948c513-a52d-4b21-aeee-70e6a8c4b79d.png)
 

From the above image we found that there are no outliers values present in our dataset. So we can skip handling of outliers values step.

Train Test Split

Now let’s split the Dataset into train and test sets

Changes: first split the dataset into x and y and then split the data set
Here x and y variables are created. On x variable, df is passed with dropping the target variable. And my target variable is passed. For splitting training and testing data we are using the train_test_split() function from sklearn. As parameters, we are passing x, y, test_size, random_state.

 
![p19](https://user-images.githubusercontent.com/111130870/219395991-77249e24-9d26-4cea-aa69-a19af5ebb9eb.png)

Oversampling Technique
 
![p20](https://user-images.githubusercontent.com/111130870/219396037-4d54183f-6b1c-4999-831f-e47a7501b326.png)

The above is using the Oversampling technique.why because my data is imbalanced that is the reason i'm importing the Oversampling.
 Oversampling is a technique to balance uneven datasets by keeping all of the data in the majority class and increasing the size of the minority class. It is one of several techniques data scientists can use to extract more accurate information from originally imbalanced datasets.
Model Building
Now our data is cleaned and it’s time to build the model. We can train our data on different algorithms. For this project we are applying four classification algorithms. The best model is saved based on its performance. 

SVC Model

A function named SVC is created and train and test data are passed as the parameters. Inside the function, the SupportVectorClassifier algorithm is initialised and training data is passed to the model with the .fit() function. Test data is predicted with .predict() function and saved in a new variable. For evaluating the model, a confusion matrix and classification report is done.
 
![p21](https://user-images.githubusercontent.com/111130870/219396101-7ec4bfb1-2d64-4b9b-8d86-126ce1f5a419.png)

Random Forest Model

A function named randomForest is created and train and test data are passed as the parameters. Inside the function, the RandomForestClassifier algorithm is initialised and training data is passed to the model with the .fit() function. Test data is predicted with the .predict() function and saved in a new variable. For evaluating the model, a confusion matrix and classification report is done.

 ![p22](https://user-images.githubusercontent.com/111130870/219396153-1b321439-7c80-427d-a7f9-fb9833db2aaf.png)


Decision Tree Model

A function named decisionTree is created and train and test data are passed as the parameters. Inside the function, DecisionTreeClassifier algorithm is initialised and training data is passed to the model with the .fit() function. Test data is predicted with .predict() function and saved in a new variable. For evaluating the model, a confusion matrix and classification report is done.

 ![p23](https://user-images.githubusercontent.com/111130870/219396173-5a3e4c1e-1bec-4e43-a61a-ac6de9a2fcb8.png)


Extra Tree Classifier Model

A function named Extra Tree is created and train and test data are passed as the parameters. Inside the function, the Extra Tree Classifier algorithm is initialised and training data is passed to the model with the fit() function. Test data is predicted with predict() function and saved in a new variable. For evaluating the model, confusion matrix and classification report is done

 ![p24](https://user-images.githubusercontent.com/111130870/219396198-1521096a-89e6-47af-88bb-f7d3357b0903.png)

Parameter Tuning

Hyper parameters

Hyperparameters are the variables that the user specifies usually while building the Machine Learning model. thus, hyperparameters are specified before specifying the parameters or we can say that hyperparameters are used to evaluate optimal parameters of the model
--For example, max depth in Random Forest Algorithms, k in KNN Classifier.
Grid Search CV
1. Search process in sequential order
2. iterate through all combinations
3. computationally cost
4. chances of overfit

Grid Search CV
Grid Search CV
1. Search process in sequential order
2. iterate through all combinations
3. computationally cost
4. chances of overfit
   
![p25](https://user-images.githubusercontent.com/111130870/219396295-8f512032-e469-4608-8406-152a8cca1b2c.png)
![p26](https://user-images.githubusercontent.com/111130870/219396673-3c4de3e2-3c38-4a54-9476-ef43b01bae12.png)
![p27](https://user-images.githubusercontent.com/111130870/219396749-2bb178a0-ce0f-4eb2-b45c-f7fd36c5e064.png)
![p28](https://user-images.githubusercontent.com/111130870/219396760-4e170873-2ce0-4885-86b1-757bb052c311.png)
![p29](https://user-images.githubusercontent.com/111130870/219396768-6f66dc12-67b9-41db-a499-cd9890ebfa86.png)
![p30](https://user-images.githubusercontent.com/111130870/219396779-1c830d01-bdcd-4b4b-80bd-147002389a00.png)
![p31](https://user-images.githubusercontent.com/111130870/219396791-131e891c-5648-4751-a596-b79de9ef30a9.png)
![p32](https://user-images.githubusercontent.com/111130870/219396802-8915db69-03a0-499d-affa-eb383e4a3a83.png)

Now let’s see the performance of all the models and save the best model

Comparison Of Models
For comparing the above four models, the compare Model function is defined.
 
After calling the function, the results of models are displayed as output. From the four models, the svc1 is performing well. From the below image, we can see the accuracy of the model is 94% accuracy.

 ![p33](https://user-images.githubusercontent.com/111130870/219396900-8085e6c1-2dd8-49ec-b36e-8d9135252ed2.png)

Evaluation Of The Model & Save The Model
From sklearn, accuracy score is used to evaluate the score of the model. On the parameters, we have given svc1 (model name), x, y, cv (as 5 folds). Our model is performing well. So, we are saving the model is svc1 by pickle.dump()..
 
 ![p34](https://user-images.githubusercontent.com/111130870/219397095-a4c0408f-ae68-42ce-b634-8d34ab4a12f2.png)

Application Building
In this section, we will be building a web application that is integrated to the model we built. A UI is provided for the uses where he has to enter the values for predictions. The enter values are given to the saved model and prediction is showcased on the UI.
This section has the following tasks
Building HTML Pages
Building server side script

Building Html Pages
For this project create three HTML files namely
•	home.html
•	predict.html
•	submit.html
and save them in the templates folder.

![p36](https://user-images.githubusercontent.com/111130870/219397167-5220412e-88a5-4e42-939e-e0a03411643f.png)
![p37](https://user-images.githubusercontent.com/111130870/219397176-622ec9fe-eae3-4fbe-9df4-351237c56c72.png)
![p38](https://user-images.githubusercontent.com/111130870/219397232-6d0ef810-edef-458f-8a8b-81410c149034.png)

Let’s see how our home.html page looks like:
 
Now when you click on predict button from top right corner you will get redirected to predict.html
Let's look how our predict.html file looks like:
 
Now when you click on submit button from left bottom corner you will get redirected to submit.html
Let's look how our submit.html file looks like:
 




Build Python Code
Import the libraries
 ![p39](https://user-images.githubusercontent.com/111130870/219397366-5dbba194-205f-4ec2-a031-4e9e8d044b0e.png)

Load the saved model. Importing the flask module in the project is mandatory. An object of Flask class is our WSGI application. Flask constructor takes the name of the current module (__name__) as argument.
 ![p40](https://user-images.githubusercontent.com/111130870/219397402-e0a3eb0b-5b92-497f-ad65-43551fad40b7.png)

Render HTML page:
 ![p41](https://user-images.githubusercontent.com/111130870/219397472-a1bd1bd1-7ba5-4db1-89d1-995136a0027f.png)

Here we will be using a declared constructor to route to the HTML page which we have created earlier.
 ![p42](https://user-images.githubusercontent.com/111130870/219397483-4ee8ac69-c8a5-46f4-9fba-4b843df4b656.png)

In the above example, ‘/’ URL is bound with the home.html function. Hence, when the home page of the web server is opened in the browser, the html page will be rendered. Whenever you enter the values from the html page the values can be retrieved using POST Method.
Retrieves the value from UI:
 ![p43](https://user-images.githubusercontent.com/111130870/219397493-f4e43f7a-44a2-4bf1-8b94-dbf8959a6ba2.png)


Here we are routing our app to predict() function. This function retrieves all the values from the HTML page using Post request. That is stored in an array. This array is passed to the model.predict() function. This function returns the prediction. And this prediction value will be rendered to the text that we have mentioned in the submit.html page earlier.
Main Function:
 ![p44](https://user-images.githubusercontent.com/111130870/219397507-d71adfbb-7dcb-44b9-b8ea-1a61bb0c62fb.png)


Run The Application
•	Open anaconda prompt from the start menu
•	Navigate to the folder where your python script is.
•	Now type “python app.py” command
•	Navigate to the localhost where you can view your web page.
•	Click on the predict button from the top right corner, enter the inputs, click on the submit button, and see the result/prediction on the web.



Output screenshots:

![p45](https://user-images.githubusercontent.com/111130870/219397725-9f4f58d9-0239-46e1-9a85-c3eb72a9e371.png)
![p46](https://user-images.githubusercontent.com/111130870/219397738-0c2f6d12-2dd0-4e09-9a95-9a6f8a480258.png)
![p47](https://user-images.githubusercontent.com/111130870/219397742-1ec9a984-e9d6-40a3-a1bd-ed06df435ebf.png)
![p48](https://user-images.githubusercontent.com/111130870/219397745-40e15cde-98e5-46a4-ae49-5b9fb7c2c90a.png)
![p49](https://user-images.githubusercontent.com/111130870/219397749-c20905c2-491b-4071-b295-3c85cf4d7842.png)
![p50](https://user-images.githubusercontent.com/111130870/219397755-481482d4-50e1-425e-b85f-11861f4e0a0a.png)



 


 
 





 
 
