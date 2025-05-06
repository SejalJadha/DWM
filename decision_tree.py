import pandas as pd
import math

# Load dataset
df = pd.read_excel(r'C:\Users\Sejal\OneDrive\Desktop\DWM prac\Electronics.xlsx')

# Drop R_Id if present
if 'R_Id' in df.columns:
    df = df.drop(columns=['R_Id'])

# Target column (assume last)
target_col = df.columns[-1]
features = [col for col in df.columns if col != target_col]

# Function to compute entropy
def entropy(data):
    total = len(data)
    if total == 0:
        return 0
    counts = data[target_col].value_counts()
    ent = 0
    for c in counts:
        p = c / total
        ent -= p * math.log2(p)
    return ent

# Function to compute information gain
def info_gain(data, feature):
    total_entropy = entropy(data)
    values = data[feature].unique()
    weighted_entropy = 0
    for val in values:
        subset = data[data[feature] == val]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset)
    return total_entropy - weighted_entropy

# Recursive tree builder
def build_tree(data, features, depth=0):
    labels = data[target_col].unique()

    # If only one class remains
    if len(labels) == 1:
        return labels[0]

    # If no features left
    if not features:
        return data[target_col].mode()[0]

    # Find best feature
    gains = {f: info_gain(data, f) for f in features}
    best_feature = max(gains, key=gains.get)
    tree = {best_feature: {}}

    for val in data[best_feature].unique():
        subset = data[data[best_feature] == val]
        if subset.empty:
            tree[best_feature][val] = data[target_col].mode()[0]
        else:
            remaining_features = [f for f in features if f != best_feature]
            tree[best_feature][val] = build_tree(subset, remaining_features, depth + 1)

    return tree

# Pretty print the tree
def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "→", tree)
        return
    for feature, branches in tree.items():
        for val, subtree in branches.items():
            print(f"{indent}If {feature} == '{val}':")
            print_tree(subtree, indent + "  ")

# Build and print the tree
tree = build_tree(df, features)
print("\nDecision Tree:")
print_tree(tree)
