from multiprocessing.shared_memory import SharedMemory
import sys
for i in range(10):
    try:
        name = f'pollen_worker_{i}'
        shm = SharedMemory(name=name, create=False)
        shm.unlink() 
    except:
        pass
try:
    name = f'pollen_config_shm'
    shm = SharedMemory(name=name, create=False)
    shm.unlink() 
except:
    pass
try:
    name = f'pollen_parameters'
    shm = SharedMemory(name=name, create=False)
    shm.unlink() 
except:
    pass


