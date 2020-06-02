class Silhouette:
  def __init__(self, X, labels):
    self.X = X
    self.labels = labels
    self.labels_unique = np.unique(labels)
  
  def euclidean_distance(self, x1, x2):
    distance = 0
    arr = x1 - x2
    for index in range(arr.shape[0]):
      arr[index] = math.pow(arr[index], 2)
    distance = math.sqrt(arr.sum(axis=0))
    return distance

  def get_points(self, label):
      indices = [i for i, x in enumerate(self.labels) if x == label]
      points = []
      for i in indices:
          points.append(self.X[i])
      return points
  
  def calculate_a(self, current_point, current_label):
    total_distance = 0
    points_correspond_to_label = self.get_points(current_label)
    for i in range(len(points_correspond_to_label)):
      if np.array_equal(points_correspond_to_label[i], current_point):
        del points_correspond_to_label[i]
        break
    for i in range(len(points_correspond_to_label)):
      total_distance += self.euclidean_distance(points_correspond_to_label[i], current_point)
    
    return total_distance/len(points_correspond_to_label)
  
  def calculate_b(self, current_point, current_label):
    result_arr = []
    remaining_labels = []
    for label in self.labels_unique:
      if label != current_label:
        remaining_labels.append(label)
    
    for label in remaining_labels:
      total_distance = 0
      points_correspond_to_label = self.get_points(label)
      for i in range(len(points_correspond_to_label)):
        total_distance += self.euclidean_distance(points_correspond_to_label[i], current_point)
      result_arr.append(total_distance/len(points_correspond_to_label))
    
    return min(result_arr)


  def score(self):
      s_arr = []
      for i in range(len(X)):
          a = None
          b = float('inf')
          b_arr = []

          a = self.calculate_a(self.X[i], self.labels[i])
          b = self.calculate_b(self.X[i], self.labels[i])

          s = (b - a) / max(a, b)
          s_arr.append(s)
      return np.mean(s_arr)