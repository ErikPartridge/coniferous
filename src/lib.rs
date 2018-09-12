enum Condition {
    EQ,
    LT,
    LTE,
    GT,
    GTE
}

pub struct Node<'a> {
    condition_type: Condition,
    value: f64,
    index: usize,
    left: Option<&'a Node<'a>>,
    right: Option<&'a Node<'a>>,
    category: Option<usize>
}

pub struct DecisionTree<'b> {
    root_node: &'b Node<'b>
}

impl<'b> DecisionTree<'b> {

    fn eval_node(node: &Node, matrix: &Vec<f64>) -> bool {
        let local_value = matrix[node.index];
        match node.condition_type {
            Condition::EQ => local_value == node.value,
            Condition::LT => local_value < node.value,
            Condition::LTE => local_value <= node.value,
            Condition::GT => local_value > node.value,
            Condition::GTE => local_value >= node.value
        }
    }

    pub fn predict(&self, x: Vec<f64>) -> usize {
        let mut temp = Some(self.root_node);
        while temp.is_some() {
            if DecisionTree::eval_node(temp.unwrap(), &x)  {
                //Then, go left, else go right
                if temp.unwrap().left.is_some() {
                    temp = temp.unwrap().left;
                } else {
                    return temp.unwrap().category.unwrap();
                }
            } else {
                if temp.unwrap().right.is_some() {
                    temp = temp.unwrap().right;
                } else {
                    return temp.unwrap().category.unwrap();
                }
            }
        }
        // This should hopefully never happen
        return std::usize::MAX;
    }

    pub fn fit(&self, x: Vec<Vec<f64>>, y: Vec<f64>) {
        
    }
    
}


#[test]
fn test_eval_node() {
    let gte = Node{condition_type: Condition::GTE, value: 0.0, index: 0, left: None, right: None, category: Some(3)};
    assert_eq!(DecisionTree::eval_node(&gte, &vec![0.01]), true);
}

#[test]
fn test_predict() {
    let gte = Node{condition_type: Condition::GTE, value: 0.0, index: 0, left: None, right: None, category: Some(3)};
    let dt = DecisionTree{root_node: &gte};
    assert_eq!(dt.predict(vec![0.01]), 3);
}