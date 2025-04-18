import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.offline as offline

offline.init_notebook_mode(connected=True)

trace = go.Scatter (x = [15, 18, 21, 25],

                    y = [100, 400, 300, 200],

                    mode = 'markers')

data = [trace]
#offline.plot(data)

# 3D
z = [25, 100, 75, 50]
trace = go.Scatter (x = [15, 18, 21, 25],

                    y = [100, 400, 300, 200],

                    mode = 'markers',

                    marker = dict(size = z)
                   )

data = [trace]
#offline.plot(data)

#4D
i = [5, 6, 8, 4]
trace = go.Scatter (x = [15, 18, 21, 25],

                    y = [100, 400, 300, 200],

                    mode = 'markers',

                    marker = dict(size = z,
                                  color = i,
                                  colorscale = 'Portland',
                                  showscale = True)
                   )

marker = dict(size = z,
              color = i,
              colorscale = 'Portland',
              showscale = True)
data = [trace]
#offline.plot(data)


housing_data = pd.read_csv('datasets/housing.csv')
housing_data = housing_data.sample(frac=0.07).reset_index(drop=True)
housing_data['ocean_proximity'].unique()
trace = go.Scatter(x = housing_data['median_income'],
                   y = housing_data['median_house_value'],

                   mode = 'markers',

                   marker = dict(
                                 size = housing_data['total_rooms'],
                                 sizeref = 500,

                                 color = housing_data['housing_median_age'],
                                 colorscale = 'Jet',
                                 showscale = True))
data = [trace]

layout = go.Layout(height = 600,
                width = 900,

                title = 'Housing Data',
                hovermode = 'closest')
fig = go.Figure(data = data,
                layout = layout)

offline.plot(fig)
