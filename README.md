# Leveraging on Counterfactuals for Prescriptive Process Analytics


Prescriptive process analytics aims to provide actionable recommendations for process instances that are predicted to fall short of achieving satisfactory outcomes.
These recommendations typically focus on assigning activities to specific resources. Given that processes may involve hundreds of resources, and hence brute-force approaches to evaluate all possible activity-resource combinations are computationally infeasible.
Current state-of-the-art techniques conversely adopt a sequential approach that selects the most suitable activity and then allocates it to one of the suitable resources: this is inherently sub-optimal.
This paper leverages counterfactual generation techniques to formulate recommendations. Counterfactual-based methods offer innovative strategies that efficiently converge to highly effective interventions. Experimental evaluations conducted on several real-world event logs demonstrate that our counterfactual-based technique outperforms a baseline approach in which the best activity is chosen and then assigned to one of the suitable resources.

