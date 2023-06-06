import re, os

def find_floats(string):
    return re.findall(r'[\d\.\d]+',string)

times = [0.01, 0.1, 1, 3, 4, 5, 6, 7, 10, 16]
sumdeg = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

for time in times:
    #Current folder
    folder_name = f"answers-{time}-no-partition"
    file_names = os.listdir(folder_name)

    print(time)

    for file_name in file_names:
        a, b = find_floats(file_name)
        a = int(float(a))
        b = int(float(b))
        sum = a+b
        print(a, b, sum)
        # Open the file
        content = open(f"{folder_name}/{file_name}", "r")
        
        # Write to correct new file
        new_file_name = f"answers-{time}-{sum}"
        f = open(f"files-for-plot-report/{new_file_name}", "a")
        f.write(content.read() + "\n")
        f.close()
        content.close()
    