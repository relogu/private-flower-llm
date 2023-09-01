from multiprocessing.shared_memory import SharedMemory

from pure_sh_node_manager import POLLEN_CONFIG_SHM, POLLEN_PARAMETERS_SHM, POLLEN_WORKER_SHM

for i in range(100):
    try:
        # name = f'pollen_worker_{i}'
        name = POLLEN_WORKER_SHM+f'{i}'
        shm = SharedMemory(name=name, create=False)
        shm.unlink() 
    except:
        pass
try:
    # name = f'pollen_config_shm'
    name = POLLEN_CONFIG_SHM
    shm = SharedMemory(name=name, create=False)
    shm.unlink() 
except:
    pass
try:
    # name = f'pollen_parameters'
    name = POLLEN_PARAMETERS_SHM
    shm = SharedMemory(name=name, create=False)
    shm.unlink() 
except:
    pass


