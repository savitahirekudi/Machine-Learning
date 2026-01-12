# import pandas as pd

# df=pd.read_csv('university_records.csv')
# print(df.to_string())

# import pandas as pd

# df=pd.read_csv('sample.xlsx')
# print(df.to_string())

# import numpy as np
# import time
# SIZE=100000
# L1=range(SIZE)
# L2=range(SIZE)
# A1=np.arange(SIZE)
# A2=np.arange(SIZE)
# start=time.time()
# result=[(x,y) for x,y in zip(L1,L2)]
# print((time.time()-start)*1000)
# start=time.time()
# result=A1+A2
# print((time.time()-start)*1000)

# import pandas as pd
# import numpy as np
# dict={'First Score':[100,90,np.nan,95],
#       'Second Score':[30,45,56,np.nan],
#       'Third Score':[np.nan,40,80,98]}
# df=pd.DataFrame(dict)
# print(df.isnull

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files (make sure they exist in the same folder)
df1 = pd.read_csv('df1.csv', index_col=0)
df2 = pd.read_csv('df2.csv')

# Inspect the data
print(df1.head())
print(df2.info())

# Plot as bar chart (only numeric columns will be plotted)
df2.plot.bar()
df1.plot.scatter(x ='A', y ='B')
df1.plot(style=['-', '--', '-.', ':'], title='Line Plot with Different Styles', xlabel='Index', ylabel='Values', grid=True)
# Show the plot
plt.show()




