☕ Bean Matched: Behavioral Coffee Roast Prediction
Bean Matched is an advanced classification engine built to predict coffee roast preferences (Light, Medium, Dark). By leveraging the "Great Coffee Survey" dataset by James Hoffmann, this project moves beyond demographic heuristics into behavioral feature engineering and cost-sensitive learning.

📊 The Dataset: "The Great Coffee Survey"
This project utilizes the comprehensive dataset curated by world-renowned coffee expert James Hoffmann. It captures the habits, preferences, and sensory perceptions of over 4000+ coffee drinkers globally.
Source: https://www.kaggle.com/datasets/datalab351/great-american-coffee-taste-test
Key Variables: Extraction methods, additive habits, specialized equipment usage, and subjective expertise levels.

🛠️ Technical Architecture & Implementation1. Feature Engineering: 
Interaction TermsWe developed interaction features to capture non-linear relationships between a user's technical knowledge and their physical consumption habits.
Coffee Maturity ($M$): Encodes the intersection of biological age and domain expertise:
$$M = Age_{encoded} \times Expertise_{scalar}$$
Expert Intensity ($I$): A proxy for "Heavy User" segmentation, weighting daily volume by technical knowledge:
$$I = Cups_{per\_day} \times Expertise_{scalar}$$
Sweetened Behavior ($S$): A composite feature quantifying the "Masking Effect"—users who add milk, sugar, or syrup often prefer roasts with higher soluble solids (Medium/Dark) to maintain flavor balance:
$$S = \sum (\text{adds\_milk}, \text{adds\_sugar}, \text{adds\_syrup})$$

Model Pipeline: XGBoost Modified
The core challenge was a severe Class Imbalance (Majority Class: Light Roast). A standard objective function led to a "Accuracy Paradox" where the model ignored minority classes to maximize global hits.
Objective Function: multi:softprob
Hyperparameter Strategy: Adjusted scale_pos_weight and custom class weights to penalize the misclassification of Medium and Dark roasts.
Evaluation Metric: Prioritized Macro-Averaged F1-Score over Accuracy to ensure the model's reliability across all labels.

📈 Results & Metrics :
Through iterative tuning, we shifted the model from a biased baseline to a balanced recommendation engine:
Metric        Baseline (Logistic)  XGBoost (Tuned)  XGBoost (Modified)
Accuracy         0.618                0.588               0.515
Macro F1-Score   0.511                0.365               0.433
Class 1 Recall   0.12                 0.05                0.45
Technical Takeaway: While global accuracy decreased by 10%, the Recall for Medium Roast increased by 800%, transforming the model from a "Light Roast" detector into a true 3-class classifier.
