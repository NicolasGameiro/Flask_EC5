import plotly.figure_factory as ff

df = [dict(Task="Calcul de tenue R387-1A", Start='2022-06-25', Finish='2022-06-26', Resource='Complete'),
      dict(Task="note argumentaire", Start='2022-06-15', Finish='2022-06-17', Resource='Incomplete'),
      dict(Task="Job-2", Start='2022-06-17', Finish='2022-06-18', Resource='Not Started'),
      dict(Task="Job-2", Start='2022-06-18', Finish='2022-06-23', Resource='Complete'),
      dict(Task="Job-3", Start='2022-06-10', Finish='2022-06-20', Resource='Not Started'),
      dict(Task="Job-4", Start='2022-06-14', Finish='2022-06-25', Resource='Complete')]

colors = {'Not Started': 'rgb(220, 0, 0)',
          'Incomplete': (1, 0.9, 0.16),
          'Complete': 'rgb(0, 255, 100)'}

fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True,
                        showgrid_x=True,
                        group_tasks=True)
fig.show()