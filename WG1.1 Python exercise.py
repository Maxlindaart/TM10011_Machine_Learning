import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

data = pd.read_csv("datasets.csv")

dataset = set(data['dataset'])

print('The number of datasets which the file datasets contains is', len(dataset))
print('The names of the datasets are:', dataset)

for a in dataset:
    subset = data[data['dataset'] == a]
    print('The statistics for dataset', a)
    print(subset.describe())

sns.violinplot(x='dataset', y='x', data=data)

plt.xlabel('Dataset')
plt.ylabel('X-coordinate')
plt.title('Violin plot of x-coordinates per dataset')

plt.show()

sns.violinplot(x='dataset', y='y', data=data)

plt.xlabel('Dataset')
plt.ylabel('Y-coordinate')
plt.title('Violin plot of y-coordinates per dataset')

plt.show()

datasets = list(dataset)

for b in datasets:
    subset = data[data['dataset'] == b]
    correlation = subset['x'].corr(subset['y'])
    cov_matrix = subset[['x','y']].cov() 
    print(f"Correlation between x and y for dataset '{b}': {correlation}")
    print(f"Covariance matrix for dataset '{b}':")
    print(cov_matrix)

    slope, intercept, r_value, p_value, std_err = linregress(subset['x'], subset['y'])
    print(f"Linear regression for dataset '{b}':")
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")
    print(f"R-value: {r_value}")

g = sns.FacetGrid(data, col="dataset", col_wrap=4, height=4, sharex=False, sharey=False)
g.map_dataframe(sns.scatterplot, x="x", y="y")
g.set_axis_labels("X-coordinate", "Y-coordinate")
g.set_titles(col_template="{col_name}")
plt.tight_layout()
plt.show()

sns.lmplot(
    data=data,
    x="x",
    y="y",
    col="dataset",        
    col_wrap=4,           
    height=4,             
    sharex=False,         
    sharey=False,         
    scatter_kws={"s": 20}, 
    line_kws={"color": "red"} 
)

plt.tight_layout()
plt.show()