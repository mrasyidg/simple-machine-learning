from CounterTool import CounterTool

class Leaf:
    
    
    def __init__(self, rows):
        self.predictions = CounterTool.class_counts(rows)
    
    @staticmethod
    def print_leaf(counts):
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        return probs