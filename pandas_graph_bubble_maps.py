import pandas as pd
import plotly.graph_objs as go
import plotly.offline as offline

offline.init_notebook_mode(connected=True)

trace = dict (type = 'scattergeo',

              lon = [-97.92, 0, 78.8],
              lat = [39.3, 0 , 21.76],

              marker = dict(size = 10),
              mode = 'markers',
             )

data = [trace]

layout = dict(showlegend = False,
           geo = dict(showland = True)
          )

fig = dict(data = data,
         layout = layout)

# offline.plot(fig)

housing_data = pd.read_csv('datasets/housing.csv')
housing_data = housing_data.sample(frac=0.1).reset_index(drop=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
housing_data['ocean_proximity_labels'] = le.fit_transform(housing_data['ocean_proximity'])

trace = dict(type = 'scattergeo',

             lat = housing_data['latitude'],
             lon = housing_data['longitude'],

             marker = dict (size = housing_data['median_house_value']/1000,
                            sizemode = 'area',
                            color = housing_data['ocean_proximity_labels'],
                            colorscale = 'Portland',
                            showscale = True),

             mode = 'markers')

data = [trace]

layout = dict(showlegend = False,
           geo = dict(showland = True,
                      landcolor = 'yellow'))
fig = dict(data = data,
         layout = layout)

offline.plot(fig)
