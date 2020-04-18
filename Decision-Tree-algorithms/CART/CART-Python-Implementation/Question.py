class Question:
    
    
    # Question Class
    def __init__(self, column, value):
        self.column = column
        self.value = value

    @staticmethod    
    def is_numeric(value):
        # To test if a value is numeric.
        return isinstance(value, int) or isinstance(value, float)
    
    def match(self, example):
        # Compare the feature value in an example to the feature value in this question.
        val = example[self.column]
        if Question.is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        """
        Fill this header. 
        e.g. 
        header = ['age', 'sex', 'total_bilirubin', 'direct_bilirubin', 'alkaline', 'alamine', 'aspartate', 'total_protein', 'albumin', 'A/G Ratio', 'label']
        
        or you can also modified it and fill the header at runtime.
        e.g. (Runtime):
        >>> header = ['age', 'sex', 'total_bilirubin', 'direct_bilirubin', 'alkaline', 'alamine', 'aspartate', 'total_protein', 'albumin', 'A/G Ratio', 'label']
        """

        # You can change the header according to your need, but for this example we'll use the ILPD's header, so this part is currently still hard coded.
        header = ['age', 'sex', 'total_bilirubin', 'direct_bilirubin', 'alkaline', 'alamine', 'aspartate', 'total_protein', 'albumin', 'A/G Ratio', 'label']
        # This is just a helper method to print the question in a readable format.
        condition = "=="
        if Question.is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))