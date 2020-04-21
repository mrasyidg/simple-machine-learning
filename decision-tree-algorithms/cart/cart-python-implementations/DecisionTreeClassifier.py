from Leaf import Leaf
from Question import Question
from DecisionNode import Decision_Node
from CounterTool import CounterTool

class DecisionTreeClassifier:
    

    def __init__(self, rows):
        self.tree = DecisionTreeClassifier.build_tree(rows)
    

    @staticmethod
    def unique_vals(rows, col):
        # Used to find the unique values for "a" column in a dataset
        return set([row[col] for row in rows])
    
    @staticmethod
    def unique_label(rows):
        # Used to find the unique values for classification_label, note that there is only one column for label. # Rasyid
        return set([row for row in rows])
        
    @staticmethod    
    def partition(rows, question):
        # Used to split a dataset into true set and false set.
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows
    
    @staticmethod   
    def gini(rows):
        # Used to count the impurity.
        counts = CounterTool.class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl**2
        return impurity
    
    @staticmethod
    def info_gain(left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * DecisionTreeClassifier.gini(left) - (1 - p) * DecisionTreeClassifier.gini(right)
    
    @staticmethod
    def find_best_split(rows):
        best_gain = 0  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        current_uncertainty = DecisionTreeClassifier.gini(rows)
        n_features = len(rows[0]) - 1  # number of columns

        for col in range(n_features):  # for each feature

            values = set([row[col] for row in rows])  # unique values in the column

            for val in values:  # for each value

                question = Question(col, val)

                # try splitting the dataset
                true_rows, false_rows = DecisionTreeClassifier.partition(rows, question)

                # Skip this split if it doesn't divide the dataset.
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                # Calculate the information gain from this split
                gain = DecisionTreeClassifier.info_gain(true_rows, false_rows, current_uncertainty)

                if gain >= best_gain:
                    best_gain, best_question = gain, question
        return best_gain, best_question
    
    @staticmethod
    def build_tree(rows):
        gain, question = DecisionTreeClassifier.find_best_split(rows)

        if gain == 0:
            return Leaf(rows)

        # If we reach here, we have found a useful feature / value to partition on.
        true_rows, false_rows = DecisionTreeClassifier.partition(rows, question)

        # Recursively build the true branch.
        true_branch = DecisionTreeClassifier.build_tree(true_rows)

        # Recursively build the false branch.
        false_branch = DecisionTreeClassifier.build_tree(false_rows)

        return Decision_Node(question, true_branch, false_branch)