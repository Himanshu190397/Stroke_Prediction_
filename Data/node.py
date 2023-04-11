
# Node class taken from provided reference https://www.tutorialspoint.com/python_data_structure/python_binary_tree.htm
class Node:
    def __init__(self):
      self.left = None
      self.right = None
  
      #column that we are splitting on
      self.f_column = None
      self.is_leaf = False
      
      self.probabilities = None
      
            
    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
        print(f"feature {self.f_column}"),
        if self.right:
            self.right.PrintTree()
    
    def PreorderTraversal(self, root):
        res = []
        if root:
            res.append(root.data)
            res = res + self.PreorderTraversal(root.left)
            res = res + self.PreorderTraversal(root.right)
        return res