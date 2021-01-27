import matplotlib.pyplot as plt
import numpy as np
import re
if __name__ == '__main__':

    slice_len = 10
    fo = open("./Result020210115.txt","r+")
    count = 0
    ans = np.array([0])
    for line in fo:
        searchObj = re.search(r'\\^\d+(\.\d+)?$\\', line)
        loc = line.find(',')
        if loc is -1:
            continue
        suc = float(line[0:loc])
        ans = np.append(ans, suc)
        count += 1
    
    ans = ans[1:]
    
    fo.close()

    start = 0
    no_attack_global = ans[start:start + slice_len]

    start += slice_len
    Converge_attack_global = ans[start:start + slice_len]
    
    start += slice_len
    Converge_attack_mal = ans[start:start + slice_len]

    start += slice_len
    filtered_attack_global = ans[start:start + slice_len]

    start += slice_len
    filtered_attack_mal = ans[start:start + slice_len]

    iteration = np.linspace(1,10,10)
    plt.plot(iteration, no_attack_global,'r',label = 'no_attack_global')
  
    plt.plot(iteration, Converge_attack_global,label='converge_attack_global')
    plt.plot(iteration, Converge_attack_mal,label='converge_attack_malagent')
    plt.plot(iteration, filtered_attack_global,label='filtered_attack_global')
    plt.plot(iteration, filtered_attack_mal,label='filtered_attack_malagent')
    
    plt.legend()

    plt.show()
