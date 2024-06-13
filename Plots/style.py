from matplotlib import pyplot as plt

#plt.rcParams['font.size'] = 8
#plt.rcParams['svg.fonttype'] = 'none'
#plt.rcParams['mathtext.fontset'] = 'custom'

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

def get_figsize(name: str) -> tuple:
    return fig_dict.get(name)
    
half_fig = (3, 2)  # Adjust as needed for your layout
figsize = (6, 4)  # Adjust as needed for your layout

fig_dict = {
    'fig': figsize,
    'half_fig': half_fig
}