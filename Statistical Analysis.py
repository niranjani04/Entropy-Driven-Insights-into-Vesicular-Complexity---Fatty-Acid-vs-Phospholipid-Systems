import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, entropy
from sklearn.decomposition import PCA
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

# Set environment order and color palette
env_order = ['Buffer', 'Vesicle', 'Oleic']
env_palette = {'Buffer': 'blue', 'Vesicle': 'green', 'Oleic': 'orange'}

def analyze_all(df):
    results = {}

    df['Environment'] = pd.Categorical(df['Environment'], categories=env_order, ordered=True)

    # PCA analysis
    print("\n=== PCA Clustering ===")
    pivot = df.pivot_table(index=['Ratio', 'Environment'],
                          columns='Product',
                          values='Concentration',
                          aggfunc='mean').fillna(0)
    
    pivot = pivot.reindex(index=sorted(pivot.index, key=lambda x: (env_order.index(x[1]), x[0])))
    X = StandardScaler().fit_transform(pivot)
    pca = PCA(n_components=2).fit(X)
    scores = pca.transform(X)
    plt.figure(figsize=(9,8))
    envs = [x[1] for x in pivot.index]
    ratios = [x[0] for x in pivot.index]
    sns.scatterplot(x=scores[:,0], y=scores[:,1],
                   hue=envs, style=ratios, s=150,
                   hue_order=env_order, palette=env_palette)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title("PCA of Product Profiles")
    plt.legend(bbox_to_anchor=(1.5,1))
    plt.show()
    print(f"Explained Variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

    # Diversity analysis
    print("\n=== Diversity Metrics ===")
    diversity_results = []
    for (ratio, env), group in df.groupby(['Ratio', 'Environment']):
        concentrations = group.groupby('Product')['Concentration'].mean().values
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
            'Num_Products': len(concentrations)
        })
    results['Diversity'] = pd.DataFrame(diversity_results)
    
    # Marker and size dictionaries
    markers = {'1:1': 'o', '1:2': 's', '1:3': 'D', '2:1': '^', '2:2': 'v', '2:3': 'P', '3:1': '*', '3:2': 'p', '3:3': 'X'}
    sizes = {'1:1': 80, '1:2': 70, '1:3': 60, '2:1': 100, '2:2': 100, '2:3': 100, '3:1': 150, '3:2': 120, '3:3': 80}
    
    if not results['Diversity'].empty:
        plt.figure(figsize=(10, 7))
        # Plot each environment-ratio combination separately
        for env in env_order:
            env_data = results['Diversity'][results['Diversity']['Environment'] == env]
            for ratio in env_data['Ratio'].unique():
                sub = env_data[env_data['Ratio'] == ratio]
                plt.scatter(sub['Shannon'], sub['Gini'],
                            marker=markers.get(str(ratio)),
                            color=env_palette[env],
                            s=sizes.get(str(ratio)),
                           )
        plt.title("Diversity vs Selectivity by Environment and Ratio")
        plt.xlabel("Shannon Entropy (Higher = More Diverse)")
        plt.ylabel("Gini Coefficient (Higher = More Selective)")
        plt.grid(True, linestyle='--', alpha=0.3)
        
        env_handles = [
            mpatches.Patch(color=env_palette[e], label=e) for e in env_order
        ]
        ratio_handles = [
            mlines.Line2D([], [], color='black', marker=markers[r], linestyle='None',
                          markersize=10, label=f'Ratio {r}') 
            for r in markers.keys()
        ]
        first_legend = plt.legend(handles=env_handles, title="Environment", loc='upper right')
        plt.gca().add_artist(first_legend)
        plt.legend(handles=ratio_handles, title="Ratio", loc='lower left')
        
        plt.tight_layout()
        plt.show()
    else:
        print("No valid data for diversity analysis")

    # Two-way ANOVA
    print("\n=== Two-Way ANOVA ===")
    anova_results = []
    for product in df['Product'].unique():
        try:
            model = ols(f'Concentration ~ C(Ratio) + C(Environment) + C(Ratio):C(Environment)',
                       data=df[df['Product'] == product]).fit()
            anova = anova_lm(model, typ=2)
            anova['Product'] = product
            anova_results.append(anova.reset_index())
        except Exception:
            continue
    if anova_results:
        results['ANOVA'] = pd.concat(anova_results)
        print(results['ANOVA'].groupby('Product').head(3))
    else:
        print("No valid data for Q6 analysis")
        results['ANOVA'] = pd.DataFrame()

    return results

# Load and process data
df = pd.read_excel("Concentration-values.xlsx")
df['Concentration'] = pd.to_numeric(df['Concentration'], errors='coerce')
df = df.dropna(subset=['Concentration'])
df['Environment'] = pd.Categorical(df['Environment'], categories=env_order, ordered=True)
full_results = analyze_all(df)