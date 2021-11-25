import numpy as np

window_size = 10

def moving_average(x, window_size):
    ret = np.sum(x,dtype=float)
    ret = ret/window_size
    return ret
    
    

data = np.zeros(window_size)
#data = np.array([10,5,8,9,15,22,26,11,15,16])


val = moving_average(data,window_size)
print(moving_average(data,window_size))


data = np.delete(data, [0])
data = np.append(data, 99)

 

val = moving_average(data,window_size)
print(moving_average(data,window_size))