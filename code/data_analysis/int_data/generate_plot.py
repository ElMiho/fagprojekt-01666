import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline

def find_succes(times_idxes, arr, time):
    for times in enumerate(times_idxes):
        if times[1] > time:
            arr[times[0]] += 1

def find_succes_sum_degree(times_idxes, sum_degrees, matrix, time, sum_degree):
    '''
    times_indxes is cols
    sum_degrees is rows
    '''
    for sd in enumerate(sum_degrees):
        if sd[1] == sum_degree:
            find_succes(times_idxes, matrix[sd[0],:], time)
            return

# Read the file
with open("data_analysis/int_data/megafile.txt", "r") as file:
    lines = file.readlines()

times = [0.01, 0.1, 1, 3, 4, 5, 6, 7, 10, 12]
number_of_succes = [0,0,0,0,0,0,0,0,0,0]
sum_degrees = [i for i in range(2,18)]
number_of_succes_matrix = np.zeros((len(sum_degrees),len(times)))
total_each_pat = [100, 100, 200, 200, 300, 300, 400, 400, 500, 400, 400, 300, 300, 200, 200, 100]

# Process each line and convert it into a SymPy array
arrays = []


time_array = []
answer_array = []

nxt = np.zeros((len(lines),len(times)))

def parse_line(line):
    count = 0
    not_done = True
    for x in enumerate(line):
        
        if x[1] == '{' and count == 1:
            index_num_start = x[0]
            if line[(x[0]+1)] == '}':
                zero_num = 0
            else:
                zero_num = 1
                
        
        if not_done and x[1] == '}' and count == 3:
            not_done = False
            index_den_end = x[0]
            sums_string = line[index_num_start:index_den_end+1]
            number_of_commas = sums_string.count(",")
            sum_degree = number_of_commas + zero_num
        
        if x[1] == '{' and count == 3:
            index_of_time = x[0]
            time = line[(x[0]+1):(x[0]+4)]
            time = float(time)
        
        elif x[1] == '{':
            count = count + 1
            
        if x[1] == '}' and line[(x[0]+1)]=='}':
            answer = line[(index_of_time+11):(x[0])]
            break

    return time, answer, sum_degree

for line in lines:    
    time, answer, sum_degree = parse_line(line)
    
    # increment every time indes that has succes
    find_succes(times, number_of_succes, time)
    find_succes_sum_degree(times, sum_degrees, number_of_succes_matrix, time, sum_degree)
    time_array.append(time)

    if answer == "$Aborted":
        val = 1
    else:
        val = 0
        
    
percentages = 1-np.asarray(number_of_succes)/4400

# plt.scatter(times,percentages)
# plt.grid(True)
# plt.xlabel("Time (s)")
# plt.ylabel("abortion rate (%)")
# plt.savefig("megaplot.png")

print("total")
print(number_of_succes_matrix)

tCat = number_of_succes_matrix.shape[1]
dCat = number_of_succes_matrix[0]

total_each_pat

# divide by number of observations.
for idx, number in enumerate(total_each_pat):
    number_of_succes_matrix[idx, :] = 1-number_of_succes_matrix[idx, :] / number
# x = np.arange(0, number_of_succes_matrix.shape[1], 1)
# y = np.arange(0, number_of_succes_matrix.shape[0], 1)

print("percentage")
print(number_of_succes_matrix.round(2))

x = times
y = sum_degrees

X, Y = np.meshgrid(x, y)

# Flatten the matrix values for the z coordinates
Z = number_of_succes_matrix.flatten()

# Create a figure and a 3D axis
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface
ax.plot_surface(X, Y, number_of_succes_matrix, cmap='viridis')

# Set labels and title
ax.set_xlabel('Time [s]')
ax.set_ylabel('Sum degree')
ax.set_zlabel('Abortion rate')
# ax.set_title('3D Matrix Plot')

plt.figure(2)
for sd in enumerate(sum_degrees):
    row = number_of_succes_matrix[sd[0],:]
    plt.plot(times, row, label = f"Sumdegree {sd[1]}")

plt.xlabel("Time [s]")
plt.ylabel("Abortion rate")
    
plt.legend()
plt.show()


plt.figure(3)
number_of_succes = [1-(i / 4400) for i in number_of_succes]
plt.grid(zorder=1)
plt.scatter(times, number_of_succes, zorder=3)
plt.plot(times, number_of_succes, label = "The combined sum degrees", color = "C1",zorder=2)
plt.title("Abortion rate pr. time interval")
plt.xlabel("Time [s]")
plt.ylabel("Abortion rate")
plt.legend()
plt.show()
