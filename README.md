# Counterfactual-based Prescriptive Process Analytics

This repository contains the implementation of the counterfactual-based framework for prescriptive process analytics, as described in our paper **Leveraging on Counterfactuals for Prescriptive Process Analytics**.

## Abstract
Prescriptive process analytics aims to provide actionable recommendations for process instances that are predicted to fall short of achieving satisfactory outcomes.
These recommendations typically focus on assigning activities to specific resources. Given that processes may involve hundreds of resources, and hence brute-force approaches to evaluate all possible activity-resource combinations are computationally infeasible.
Current state-of-the-art techniques conversely adopt a sequential approach that selects the most suitable activity and then allocates it to one of the suitable resources: this is inherently sub-optimal.
This paper leverages counterfactual generation techniques to formulate recommendations. Counterfactual-based methods offer innovative strategies that efficiently converge to highly effective interventions. Experimental evaluations conducted on several real-life datasets demonstrate that our counterfactual-based technique outperforms a baseline approach in which the best activity is chosen and then assigned to one of the suitable resources.


## Framework
<img width="1023" alt="Counterfactuals_frameworks_" src="https://github.com/user-attachments/assets/1ca91681-f755-4724-855d-9673c8ce1038" />


## Installation
**Dependencies**

This implementation requires the following Python libraries:
```python
pip install pandas numpy catboost dice_ml
```

## Usage
1. Preprocessing Event Logs
2. Train the Total Time Oracle model
3. Generate Counterfactual-based recommendations
   Run:
```python
python run_experiment.py --case_study "bac" --reduced_threshold 0.5 --num_cfes 100
```

## License
This project is licensed under the MIT License.
