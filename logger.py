import csv
import argparse
import os
import time as T
from datetime import datetime

import psutil as pu
import torch

if torch.cuda.is_available():
    # from pynvml import *
    # nvmlInit()
    # handle = nvmlDeviceGetHandleByIndex(0)
    # gpu = nvmlDeviceGetMemoryInfo(handle)
    
    dev = torch.device('cuda:0')
    print('GPU OK!')
else:
    print('No GPU...')


def get_data(interval, pid):
    p = pu.Process(pid)
    
    time = datetime.now()
    
    cpu_percent = pu.cpu_percent(interval)
    # cpu_freq = round(pu.cpu_freq().current, 3)
    
    # mem = pu.virtual_memory()
    mem_per = round(p.memory_percent(), 3)
    mem_use = round((p.memory_info().rss) / 1024 ** 2, 3)
    mem_tot = round(pu.virtual_memory().total / 1024 ** 2, 3)
    
    swap = pu.swap_memory()
    swap_use = round((swap.total - swap.free)/1024**2, 3)
    swap_per = swap.percent
    
    disk = pu.disk_usage('./')
    disk_use = disk.used
    disk_per = disk.percent
    
    if torch.cuda.is_available():        
        gpu_mem_fre, gpu_mem_tot = torch.cuda.mem_get_info(dev)
        
        gpu_mem_fre = round(gpu_mem_fre / 1024 ** 2, 3)
        gpu_mem_tot = round(gpu_mem_tot / 1024 ** 2, 3)
        gpu_mem_use = round(gpu_mem_tot - gpu_mem_fre, 3)
        
        stats = {
            'time':time, 'cpu_percent':cpu_percent, #'cpu_freq':cpu_freq,
            'mem_per':mem_per, 'mem_use':mem_use, 'mem_tot':mem_use, 'mem_tot':mem_tot,
            'swap_per':swap_per, 'swap_use':swap_use,
            'disk_per':disk_per, 'disk_use':disk_use,
            'gpu_mem_use':gpu_mem_use, 'gpu_mem_fre':gpu_mem_fre, 'gpu_mem_tot':gpu_mem_tot
        }
    else:
        stats = {
            'time':time, 'cpu_percent':cpu_percent, #'cpu_freq':cpu_freq,
            'mem_per':mem_per, 'mem_use':mem_use, 'mem_tot':mem_use,
            'swap_per':swap_per, 'swap_use':swap_use,
            'disk_per':disk_per, 'disk_use':disk_use
            #'gpu_mem_per':gpu_mem_per, 'gpu_mem_use':gpu_mem_use, 'gpu_mem_fre':gpu_mem_fre, 'gpu_mem_tot':gpu_mem_tot
        }
    
    T.sleep(interval)
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Simple logger')

    parser.add_argument('--file', action="store", dest="file", default="log.csv")
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--pid', type=int, default=None)
    args = parser.parse_args()

    save_path = './logger/'
    save_file = os.path.join(save_path, args.file)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    print("Simple jtop logger")
    print(f"Saving log on {save_file}")
    print(f'Process PID {os.getpid()}')

    with open(save_file, 'w') as csvfile:
        stats = get_data(args.interval, args.pid)
        
        writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
        writer.writeheader()
        writer.writerow(stats)
        
        print('Logging...')
        
        while True:
            stats = get_data(args.interval, args.pid)
            writer.writerow(stats)
            #print("Log at {time}".format(time=stats['time']))
            csvfile.flush()

if __name__ == "__main__":
    main()

