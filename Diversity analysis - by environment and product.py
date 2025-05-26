import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.stats import entropy

env_order = ['Buffer', 'Vesicle', 'Oleic']
env_palette = {'Buffer': 'blue', 'Vesicle': 'green', 'Oleic': 'orange'}

def analyze_all(df):
    results = {}

    df['Environment'] = pd.Categorical(df['Environment'], categories=env_order, ordered=True)
    
    # Diversity analysis - by environment and product
    print("\n=== Diversity Metrics ===")
    diversity_results = []
    for (ratio, env), group in df.groupby(['Product', 'Environment']):
        concentrations = group.groupby('Ratio')['Concentration'].mean().values
        concentrations = concentrations[concentrations > 0]
        if len(concentrations) == 0:
            continue
        total = concentrations.sum()
        if total > 0:
            probs = concentrations / total
            shannon = entropy(probs, base=2)
        else:
            shannon = 0
        sorted_conc = np.sort(concentrations)
        n = len(sorted_conc)
        gini = (np.sum(np.abs(np.subtract.outer(sorted_conc, sorted_conc))) / (2 * n * total)) if n > 0 else 0
        diversity_results.append({
            'Ratio': ratio,
            'Environment': env,
            'Shannon': round(shannon, 2),
            'Gini': round(gini, 3),
            'Num_Products': len(concentrations),
            'Product': ','.join(group['Product'].unique())
        })
    results['Diversity'] = pd.DataFrame(diversity_results)

    # Custom marker properties
    product_list = sorted(df['Product'].unique())
    marker_list = ['o', 's', 'D', '^', 'v', 'P', '*', 'p', 'X']
    product_markers = {prod: marker_list[i % len(marker_list)] for i, prod in enumerate(product_list)}
    
    sizes = {
    'A1H1': 170,
    'A1H2': 170,
    'A1H3': 170,
    'A2H1': 170,
    'A2H2': 170,
    'A2H3': 170,
    'A3H1': 190,
    'A3H2': 170,
    'A3H3': 170
    }
    default_size = 170

    if not results['Diversity'].empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, env in enumerate(env_order):
            ax = axes[i]
            env_data = results['Diversity'][results['Diversity']['Environment'] == env]
            
            for prod in product_list:
                sub = env_data[env_data['Product'].str.contains(prod)]
                if sub.empty:
                    continue
                ax.scatter(sub['Shannon'], sub['Gini'],
                         marker=product_markers[prod],
                         color=env_palette[env],
                         s=sizes.get(prod, default_size),
                         edgecolor='black',
                         linewidth=1,
                         alpha=0.9)
            
            ax.set_xlim(2.5, 3.2)
            ax.set_ylim(0, 0.5)
            
            ax.set_title(f"{env} Environment", fontsize=14)
            ax.set_xlabel("Shannon Entropy (Higher = More Diverse)")
            ax.set_ylabel("Gini Coefficient (Higher = More Selective)")
            ax.grid(True, linestyle='--', alpha=0.3)
            
            prod_handles = [
                mlines.Line2D([], [], 
                              marker=product_markers[p],
                              markerfacecolor='white',
                              markeredgecolor='black',
                              markersize=10,
                              linestyle='None',
                              label=f'{p}') 
                for p in product_list
            ]
            ax.legend(handles=prod_handles, title="Product", 
                     loc='lower left', framealpha=0.9)
        
        plt.suptitle("Diversity vs Selectivity by Environment and Product", fontsize=16)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid data for diversity analysis")

    return results

# Load and process data
df = pd.read_excel("Concentration-values.xlsx")
df['Concentration'] = pd.to_numeric(df['Concentration'], errors='coerce')
df = df.dropna(subset=['Concentration'])
df['Environment'] = pd.Categorical(df['Environment'], categories=env_order, ordered=True)
full_results = analyze_all(df)