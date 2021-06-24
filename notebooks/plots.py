import matplotlib as mpl
import matplotlib.pylab as plt

def setup(font_size=8):
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'font.size': font_size,
    })

def save_figure(fig, name):
    plt.tight_layout()
    fig.savefig(f'{name}.pgf', format='pgf')