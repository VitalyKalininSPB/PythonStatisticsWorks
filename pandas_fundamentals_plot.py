import pandas as pd
import os

df = pd.read_pickle(os.path.join('..', 'data_frame.pickle'))

# Simplest default plot
acquisition_years = df.groupby('acquisitionYear').size()
#acquisition_years.plot()
import matplotlib.pyplot as plt
#plt.show()

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True,
                 'axes.titlepad': 20})

# Add axis labels
fig = plt.figure()
subplot = fig.add_subplot(1, 1, 1)
acquisition_years.plot(ax=subplot)
subplot.set_xlabel("Acquisition Year")
subplot.set_ylabel("Artworks Acquired")
fig.show()

# Increase ticks granularity
# Rotate X ticks
fig = plt.figure()
subplot = fig.add_subplot(1, 1, 1)
acquisition_years.plot(ax=subplot, rot=45, grid=True)
subplot.set_xlabel("Acquisition Year")
subplot.set_ylabel("Artworks Acquired")
subplot.locator_params(nbins=40, axis='x')
fig.show()

plt.show()
