from Leaf import *

class Decision_Node:
    
    
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        
    @staticmethod    
    def print_tree(node, spacing=""):
        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            print (spacing + "Predict", node.predictions)
            return

        # Print the question at this node
        print (spacing + str(node.question))

        # Call this function recursively on the true branch
        print (spacing + '--> True:')
        Decision_Node.print_tree(node.true_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print (spacing + '--> False:')
        Decision_Node.print_tree(node.false_branch, spacing + "  ")

    @staticmethod  
    def classify(row, node):
        if isinstance(node,Leaf):
            return node.predictions

        if node.question.match(row):
            return Decision_Node.classify(row, node.true_branch)
        else:
            return Decision_Node.classify(row, node.false_branch)
    
    @staticmethod
    def testing_result(testing_dataset, tree):
        for row in testing_dataset:
            print("Actual: %s. Predicted: %s" %
                (row[-1], Leaf.print_leaf(Decision_Node.classify(row, tree))))