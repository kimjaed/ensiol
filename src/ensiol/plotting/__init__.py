"""Figure code. Every function returns the matplotlib figure and never calls ``plt.show``."""
from .evolution import plot_evolution
from .publication import PublicationStyle, plot_evolution_publication
from .ensembles import plot_ensemble
from .comparison import ComparisonStyle, plot_comparison, accuracy_vs_cost_summary

__all__ = ["plot_evolution", "PublicationStyle", "plot_evolution_publication", "plot_ensemble",
           "ComparisonStyle", "plot_comparison", "accuracy_vs_cost_summary"]
