# Wheat Classification

## Program uses supervised learning to predict wheat species based on seed measurements of different wheat varieties.

Used algorithms for classification:
    Decision Tree
    SVM
    Random Forest

### Data preparation
Data was imported from csv file using pandas library. The data set did not require cleaning.

### Decision Tree Classifier
Decision tree classifier is a machine learning model that makes decisions by dividing input data based on a set of decision rules.
Image of decision tree drawn using matplotlib and plot_tree from sklearn:

![wheat_decision_tree](https://github.com/s22018-pj-Oliwier-Kossak/Nai/assets/73356741/8bb80a52-3514-4e84-8d41-1bcb621056da)

### Random Forest Classifier
Random forest classifier is a machine learning algorithm that combines multiple decision trees to produce more stable and accurate predictions.
The tree created as a result of voting is an abstract creation, it is not possible to draw an image of a tree. The image below is the final tree generated by the random forest:

![wheat_random_forest_last_decision_tree](https://github.com/s22018-pj-Oliwier-Kossak/Nai/assets/73356741/8d2b84f9-d14c-494e-8160-e1fdea523ea0)

### Support Vector Machine
Support vector machine is a machine learning algorithm that is used in both classification and regression problems. 
Its main goal is to find a hyperplane in a multidimensional space that best separates data belonging to different classes.

### Accuracy Of Classifiers 

![program_output](https://github.com/s22018-pj-Oliwier-Kossak/Nai/assets/73356741/b6e5a513-77ed-4961-9139-9db11d76919e)

### Bar chart for accuracy 

![wheat_bar_chart](https://github.com/s22018-pj-Oliwier-Kossak/Nai/assets/73356741/d79114ac-5559-4ec0-bffc-1bd460e623e2)


# Titanic Classification

## The program uses supervised learning to predict, based on data, whether the Titanic catastrophe passenger will survive.
Classification algorithms used:
    Decision Tree
    SVM
    Random Forest
    
### Data preparation
Data was imported from csv file using pandas library.
Most of the columns have been removed because they can interfere with the effectiveness of the model.
Deleted columns:
    PassengerId: The unique ID of the passenger. Each passenger is assigned a unique identification number.
    Name: Passenger's name and surname.
    SibSp: Number of siblings or spouses of the passenger on board.
    Parch: The number of a passenger's children or parents on board.
    Ticket: Passenger's ticket number.
    Fare: The ticket price that the passenger paid.
    Cabin: The number of the cabin in which the passenger lived. This field may contain missing values.
    Embarked: The place where the passenger boarded the ship. It can take the values ​​"C" (Cherbourg), "Q" (Queenstown) or "S" (Southampton).
Rows that do not contain the passenger's age have been deleted.

### Decision Tree Classifier

![titanic_decision_tree](https://github.com/s22018-pj-Oliwier-Kossak/Nai/assets/73356741/281f8bd2-9293-4345-8e54-f751be5a649e)

### Random Forest Classifier
The tree created as a result of voting is an abstract creation, it is not possible to draw an image of a tree. The image below is the final tree generated by the random forest:

![titanic_random_forest_last_tree](https://github.com/s22018-pj-Oliwier-Kossak/Nai/assets/73356741/d1160641-0f52-4457-9949-021402b80e14)

### Accuracy Of Classifiers 

![program_output](https://github.com/s22018-pj-Oliwier-Kossak/Nai/assets/73356741/d781745a-0778-42c6-ab12-38b0ce2cf277)

### Bar chart for accuracy 

![titanic_bar_chart](https://github.com/s22018-pj-Oliwier-Kossak/Nai/assets/73356741/5591c4be-bc43-4beb-8d32-c9f640a01857)

## Conclusions

It can be seen that despite a significant reduction in the amount of independent data in the "Titanic set", the classification model copes much better with the wheat data set. 
This is probably due to the fact that the age data in the Titanic collection varies greatly. 
In the case of Titanic, we are dealing with a binary classification, while in the wheat harvest we are dealing with a multi-class classification.
