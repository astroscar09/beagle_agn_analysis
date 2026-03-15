import matplotlib.pyplot as plt 

cobaltblue = '#2e37fe'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['font.family'] = 'serif'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

def plot_model_comparison(df, show = None, save = None):

    # Assuming `df` is the dataframe that has the bayesian evidence

    plt.figure(figsize=(8, 5))
    plt.barh(df["Model"], df["logZ"], xerr=df["logZ_err"], color="skyblue", edgecolor="black")
    plt.xlabel("log Evidence (logZ)")
    plt.title("Bayesian Evidence Comparison")
    plt.gca().invert_yaxis()  # Highest logZ at top
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300)
    
    if show:
        plt.show()