from matplotlib import pyplot as plt

#plt.rcParams['font.size'] = 8
#plt.rcParams['svg.fonttype'] = 'none'
#plt.rcParams['mathtext.fontset'] = 'custom'

plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 15
})

def get_figsize(name: str) -> tuple:
    return fig_dict.get(name)
    
half_fig = (3, 3)  # Adjust as needed for your layout
figsize = (7.5, 5)  # Adjust as needed for your layout
long_fig = (4, 13)  # Adjust as needed for your layout

fig_dict = {
    'fig': figsize,
    'half_fig': half_fig,
    'long_fig': long_fig,
}