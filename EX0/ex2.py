import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('EX0/applesOranges.csv')

def create_2D_fruit_scatte():

     plt.scatter(data['x.2'], data['x.1'], c=data['y'])
     plt.xlabel('X1')
     plt.ylabel('X2')
     plt.title('Apples and Oranges')
     plt.show()  

#create_2D_fruit_scatte()

def conditional_densities(type:str):
     if type == 'apple':
          apple_data = data[data['y'] == 0]
          plt.hist2d(apple_data['x.2'], apple_data['x.1'], bins=30)
          plt.show()
     else :
          orange_data = data[data['y'] == 1]
          plt.hist2d(orange_data['x.2'], orange_data['x.1'], bins=30)


#conditional_densities('orange')

def conditional_densities_together():
     apple_data = data[data['y'] == 0]
     orange_data = data[data['y'] == 1]

     plt.hist2d(apple_data['x.2'], apple_data['x.1'], label ='apple', cmap='Blues', bins=30, alpha=0.5)
     plt.hist2d(orange_data['x.2'], orange_data['x.1'], label='orange', cmap='Reds', bins=30, alpha=0.5)
     plt.show()

conditional_densities_together()


