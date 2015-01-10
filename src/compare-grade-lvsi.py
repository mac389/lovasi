import numpy as np
import matplotlib.pyplot as plt
import graphics as artist

from matplotlib import rcParams

rcParams['text.usetex'] = True

data = [(.77,.73),(.75,.64),(.75,.67),(.67,.67),(.74,.66),(.71,.72),(.74,.57),(.7,.68),(.72,.71),(.67,.74),(.64,.75),(.7,.7),(.7,.69)]

fig = plt.figure()
ax = fig.add_subplot(111)
x,y = zip(*data)
ax.scatter(x,y,s=10,c='k')
artist.adjust_spines(ax)
ax.set_xlabel(r'\Large \textbf{\textsc{Histologic grading}}')
ax.set_ylabel(r'\Large \textbf{\textsc{{\LARGE VLSI} grading}}')
plt.tight_layout()
plt.show()