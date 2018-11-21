import numpy as np
def transform_data(dat):
    func=lambda x : [
                             #1st Order
                             ##1 variable
                             x[0],x[1],x[2],x[3],
                             
                             #2nd Order
                             ##1 variable
                             x[0]**2,x[1]**2,x[2]**2,x[3]**2,
                             ##2 variable
                             x[0]*x[1],x[0]*x[2],x[0]*x[3],x[1]*x[2],x[1]*x[3],x[2]*x[3],
            
                             #3rd Order
                             ##1 variable
                             x[0]**3,x[1]**3,x[2]**3,x[3]**3,
                             ##2 variables
                             (x[0]**2)*x[1],(x[0]**2)*x[2],(x[0]**2)*x[3],
                             (x[1]**2)*x[0],(x[1]**2)*x[2],(x[1]**2)*x[3],
                             (x[2]**2)*x[0],(x[2]**2)*x[1],(x[2]**2)*x[3],
                             (x[3]**2)*x[0],(x[3]**2)*x[1],(x[3]**2)*x[2],
                             ##3 variables
                             x[0]*x[1]*x[2],x[0]*x[1]*x[3],x[0]*x[2]*x[3],x[1]*x[2]*x[3],
                             ]
    transformed_data = np.array(list(map(func,dat)))
    return transformed_data

data=np.load("new_mined_data_1.npy")
x=np.array([i[0] for i in  data])
modified_x=transform_data(x)

y=np.array([i[1] for i in  data])
