extern crate rand;
extern crate rayon;

use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering::Equal;
use std::collections::HashMap;
use std::sync::Arc;

pub enum Splitter {
    RANDOM,
    BEST,
}

pub struct DTParameters {
    splitter: Splitter,
    max_depth: usize,
    min_samples_split: usize,
    max_features: usize,
}

#[derive(Clone)]
pub struct Node {
    value: f64,
    index: u64,
    l: Option<Arc<Node>>,
    r: Option<Arc<Node>>,
    category: Option<u64>,
}

/// The model for a decision tree backed by the CART algorithm. A binary tree.
pub struct CARTree {
    root_node: Node,
}

/// Computes the gini impurity for a dataset, we need this for the CART algorithm.
fn gini(predicted: &Vec<u64>) -> f64 {
    if predicted.len() == 0 {
        return 1.0;
    }
    fn p_squared(count: u64, len: f64) -> f64 {
        let p = count as f64 / len;
        return p * p;
    }

    let len = predicted.len() as f64;
    let mut sorted = predicted.clone();
    sorted.par_sort();
    let mut prev = std::u64::MAX;
    let mut sum = 1.0;
    let mut local_count = 0;
    while sorted.len() != 0 {
        let temp = sorted.pop().unwrap();
        if temp != prev {
            sum -= p_squared(local_count, len);
            prev = temp;
            local_count = 0;
        }
        local_count += 1;
    }

    sum -= p_squared(local_count, len);

    return sum;
}

fn mode(numbers: &[u64]) -> u64 {
    let mut count = HashMap::new();

    for &value in numbers {
        *count.entry(value).or_insert(0) += 1;
    }

    return count
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map(|(val, _)| val)
        .expect("Cannot compute the mode of nothing");
}

type Criteria = (f64, u64);

impl CARTree {
    /// Evaluate the truthiness of the condition for a given value. Can also be thought of as, if less than criteria, go left, else right.
    ///
    fn eval_node(node: &Node, matrix: &[f64]) -> bool {
        return matrix[node.index as usize] < node.value;
    }

    fn traverse(node: &Node, matrix: &[f64]) -> u64 {
        match CARTree::eval_node(node, matrix) {
            true => match node.l {
                Some(ref n) => CARTree::traverse(&n, matrix),
                _ => return node.category.unwrap(),
            },
            false => match node.r {
                Some(ref n) => CARTree::traverse(&n, matrix),
                _ => return node.category.unwrap(),
            },
        }
    }
    /// Given a vector of x values, predict the corresponding category. Requires that the decision tree has already been fit.
    ///
    /// ### In Review
    /// Should this method potentially take a reference as opposed to ownership?
    ///
    pub fn predict(&self, x: &[f64]) -> u64 {
        return CARTree::traverse(&self.root_node, &x);
    }

    ///
    /// ```
    /// let x1 = vec![0.0, 1.0, 2.0];
    /// let x2 = vec![2.0, 1.0, 0.0];
    /// let x = vec![x1, x2];
    /// // In this case, x[0] == x1 && x[0][2] == 2.0
    /// ```
    pub fn fit(x: &[Vec<f64>], y: &[u64], params: DTParameters) -> CARTree {
        return CARTree {
            root_node: CARTree::best_criteria(x, &y, &params, 0),
        };
    }

    /// Generate potential splitting points
    fn criteria_options(x: &[Vec<f64>], params: &DTParameters) -> Vec<Criteria> {
        let potential_criteria: Vec<Vec<(f64, u64)>> = x[0]
            .par_iter()
            .enumerate()
            .filter(|_| {
                return rand::thread_rng().gen::<f64>()
                    < params.max_features as f64 / x[0].len() as f64;
            }).map(|(i, _)| {
                let mut column: Vec<(f64, u64)> = x.par_iter().map(|v| (v[i], i as u64)).collect();
                column.par_sort_by(|a, b| a.partial_cmp(b).unwrap_or(Equal));
                column.dedup();
                return column;
            }).collect();
        return potential_criteria
            .iter()
            .flat_map(|v| v.iter())
            .cloned()
            .collect();
    }

    fn eval_criteria(
        criteria: &Criteria,
        x: &[Vec<f64>],
        y: &[u64],
    ) -> (Vec<u64>, Vec<u64>, Vec<usize>, Vec<usize>) {
        let mut y1 = Vec::new();
        let mut y2 = Vec::new();
        let mut index1 = Vec::new();
        let mut index2 = Vec::new();
        let node = Node {
            value: criteria.0,
            index: criteria.1,
            l: None,
            r: None,
            category: None,
        };
        for i in 0..x.len() {
            if CARTree::eval_node(&node, &x[i]) {
                y1.push(y[i]);
                index1.push(i);
            } else {
                y2.push(y[i]);
                index2.push(i);
            }
        }
        return (y1, y2, index1, index2);
    }

    fn get_split(
        split: &(Vec<u64>, Vec<u64>, Vec<usize>, Vec<usize>),
        x: &[Vec<f64>],
        left: bool,
    ) -> Vec<Vec<f64>> {
        let indicies;
        if left {
            indicies = split.2.clone();
        } else {
            indicies = split.3.clone();
        }
        return indicies.par_iter().map(|index| x[*index].clone()).collect();
    }

    fn weighted_gini(c: Criteria, x: &[Vec<f64>], y: &[u64]) -> f64 {
        let split = CARTree::eval_criteria(&c, x, y);
        let gini_left = gini(&split.0) * (split.0.len() + 1) as f64;
        let gini_right = gini(&split.1) * (split.1.len() + 1) as f64;
        return gini_left + gini_right;
    }

    fn best_split(x: &[Vec<f64>], y: &[u64], params: &DTParameters) -> Criteria {
        match params.splitter {
            Splitter::BEST => {
                return *CARTree::criteria_options(x, params)
                    .par_iter()
                    .min_by(|a, b| {
                        let gini_a = CARTree::weighted_gini(**a, x, y);
                        let gini_b = CARTree::weighted_gini(**b, x, y);
                        return gini_a.partial_cmp(&gini_b).unwrap_or(Equal);
                    }).unwrap();
            }
            Splitter::RANDOM => {
                let options = CARTree::criteria_options(x, params);
                return options[rand::thread_rng().gen_range(0, options.len())];
            }
        }
    }

    fn best_criteria(x: &[Vec<f64>], y: &[u64], params: &DTParameters, depth: usize) -> Node {
        if depth == params.max_depth || y.len() <= params.min_samples_split {
            return Node {
                value: 0.0,
                index: 0,
                l: None,
                r: None,
                category: Some(mode(y)),
            };
        }
        let best_choice = CARTree::best_split(x, y, params);
        let split = CARTree::eval_criteria(&best_choice, x, y);
        let gini_left = gini(&split.0);
        let gini_right = gini(&split.1);
        if gini_left == 0.0 && gini_right == 0.0 {
            let left = Arc::new(Node {
                value: 0.0,
                index: 0,
                l: None,
                r: None,
                category: Some(split.0[0]),
            });
            let right = Arc::new(Node {
                value: 0.0,
                index: 0,
                l: None,
                r: None,
                category: Some(split.1[0]),
            });
            return Node {
                value: best_choice.0,
                index: best_choice.1,
                l: Some(left),
                r: Some(right),
                category: None,
            };
        } else if gini_left == 0.0 {
            let right = CARTree::best_criteria(
                &CARTree::get_split(&split, x, false),
                &split.1,
                params,
                depth + 1,
            );
            return Node {
                value: best_choice.0,
                index: best_choice.1,
                l: None,
                r: Some(Arc::new(right)),
                category: Some(split.0[0]),
            };
        } else if gini_right == 0.0 {
            let left = CARTree::best_criteria(
                &CARTree::get_split(&split, x, true),
                &split.0,
                params,
                depth + 1,
            );
            return Node {
                value: best_choice.0,
                index: best_choice.1,
                l: Some(Arc::new(left)),
                r: None,
                category: Some(split.1[0]),
            };
        } else {
            let left = CARTree::best_criteria(
                &CARTree::get_split(&split, x, true),
                &split.0,
                params,
                depth + 1,
            );
            let right = CARTree::best_criteria(
                &CARTree::get_split(&split, x, false),
                &split.1,
                params,
                depth + 1,
            );
            return Node {
                value: best_choice.0,
                index: best_choice.1,
                l: Some(Arc::new(left)),
                r: Some(Arc::new(right)),
                category: None,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_node() {
        let gte = Node {
            value: 0.0,
            index: 0,
            l: None,
            r: None,
            category: Some(3),
        };
        assert_eq!(CARTree::eval_node(&gte, &vec![0.01]), false);
    }

    #[test]
    fn test_gini() {
        let vec = vec![0, 0, 0, 1];
        assert_eq!(0.375, gini(&vec));
        let v2 = vec![0, 0];
        assert_eq!(0.0, gini(&v2));
        let mut v3 = vec![0];
        v3.pop();
        assert_eq!(1.0, gini(&v3));
    }

    #[test]
    fn test_generate_criteria() {
        let vector = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let params = DTParameters {
            splitter: Splitter::BEST,
            max_depth: 300,
            min_samples_split: 0,
            max_features: 300,
        };
        let result = CARTree::criteria_options(&vector, &params);
        assert_eq!(result.len(), 4);
        let vector2 = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 0.0]];
        let result = CARTree::criteria_options(&vector2, &params);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_best_split() {
        let vector = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 0.0]];
        let params = DTParameters {
            splitter: Splitter::BEST,
            max_depth: 300,
            min_samples_split: 0,
            max_features: 300,
        };
        let result = CARTree::best_split(&vector, &vec![0, 1, 0], &params);
        assert_eq!(result.0, 1.0);
        assert_eq!(result.1, 0);
    }

    #[test]
    fn test_best_criteria() {
        let vector = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 0.0]];
        let params = DTParameters {
            splitter: Splitter::BEST,
            max_depth: 300,
            min_samples_split: 0,
            max_features: 300,
        };
        let root_node = CARTree::best_criteria(&vector, &vec![0, 1, 0], &params, 0);
        assert!(root_node.l.is_some());
        assert!(root_node.r.is_some());
    }

    #[test]
    fn test_decision_tree() {
        let vector = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 0.0]];
        let params = DTParameters {
            splitter: Splitter::BEST,
            max_depth: 300,
            min_samples_split: 0,
            max_features: 300,
        };
        let dt = CARTree::fit(&vector, &vec![0, 1, 0], params);
        let res = dt.predict(&vec![0.0, 0.0]);
        assert_eq!(res, 0);
    }

}
