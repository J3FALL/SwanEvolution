import matplotlib.pyplot as plt
import seaborn as sns


def population_variance_boxplots(history):
    variance = [p.pop_variance for p in history]
    plt.figure(figsize=(5, 5))
    sns.boxplot(
        data=variance,
    )
    plt.show()
