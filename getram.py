import nvidia_smi
import os
import time
import psutil

Stopt = 0

def psutilvm():
# gives a single float value
#psutil.cpu_percent()
# gives an object with many fields
#psutil.virtual_memory()
# you can convert that object to a dictionary 
#dict(psutil.virtual_memory()._asdict())
# you can have the percentage of used RAM
#psutil.virtual_memory().percent
  while True:
    time.sleep(1)
    print("ava ", psutil.virtual_memory().available, "total ", psutil.virtual_memory().total,
    "used ", psutil.virtual_memory().used)
    if Stopt:
      break
79.2
# you can calculate percentage of available memory
#psutil.virtual_memory().available * 100 / psutil.virtual_memory().total


def updatestop(s):
  Stopt = s

def getGPURAM():
  nvidia_smi.nvmlInit()

  deviceCount = nvidia_smi.nvmlDeviceGetCount()
  while True:
    time.sleep(1)
    for i in range(deviceCount):
      handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
      info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
      print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
    if Stopt:
      break

  nvidia_smi.nvmlShutdown()

def getCPURAM():
  # Getting all memory using os.popen()
  while True:
    time.sleep(5)
    total_memory, used_memory, free_memory = map(
    int, os.popen('free -t -m').readlines()[-1].split()[1:])
  
    # Memory usage
    #print("RAM memory % used:", round((used_memory/total_memory) * 100, 2))
    print("RAM memory: total_memory =", total_memory, " used : ", used_memory, " free ", free_memory)
    if Stopt:
      break

def getram_cpu():
  total_memory, used_memory, free_memory = map(
    int, os.popen('free -t -m').readlines()[-1].split()[1:])
  return (total_memory, used_memory, free_memory)