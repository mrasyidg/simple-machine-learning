class CounterTool:
        def __init__(self, rows):
            self.counts = CounterTool.class_counts(rows)

        @staticmethod
        def class_counts(rows):
                # Used for dataset array. Returns a dictionary of label -> count.
                counts = {}
                for row in rows:
                    # in our dataset format, the label is always the last column
                    label = row[-1]
                    if label not in counts:
                        counts[label] = 0
                    counts[label] += 1
                return counts