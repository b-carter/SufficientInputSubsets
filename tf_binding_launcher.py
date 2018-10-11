import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor as Pool


######################
##    Params
GPUS = 7
MAX_JOBS_PER_GPU = 1
GPU_START_INDEX = 1
TASK = 'motif_occupancy'
######################


class GPUManager(object):
    def __init__(self, gpus, max_jobs_per_gpu, start_index=0):
        self.gpu_jobs_available = [max_jobs_per_gpu for _ in range(gpus)]
        self.start_index = start_index

    def get_gpu(self):
        for i in range(len(self.gpu_jobs_available)):
            if self.gpu_jobs_available[i] >= 1:
                self.gpu_jobs_available[i] -= 1
                return i + self.start_index
        return None

    def free_gpu(self, i):
        self.gpu_jobs_available[i - self.start_index] += 1

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<GPUManager: %s>' % str(self.gpu_jobs_available)


# Initialize globals
gpu_manager = GPUManager(GPUS, MAX_JOBS_PER_GPU, start_index=GPU_START_INDEX)
f_to_gpu = {}


def get_datasets(data_directory):
    datasets = next(os.walk(data_directory))[1]
    return datasets

def make_cmd(dataset, gpu, task):
    cmd = ['python tf_binding.py',
           '-dataset %s' % dataset,
           '-gpu %d' % int(gpu),
           '-log log.txt',
           '-task %s' % task,
           '-outbasedir rationale_results/motif',
           #'--train']
           '--runsis --runalternatives --comparemethods']
    return ' '.join(cmd)

def submit_job(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
    print('Submitting: ', cmd)
    f = pool.submit(subprocess.call, cmd,
                      stdout=stdout,
                      stderr=stderr,
                      shell=True)
    f.add_done_callback(add_job_callback)
    return f

def add_job_callback(f):
    gpu = f_to_gpu[f]
    print('Freeing gpu', gpu)
    gpu_manager.free_gpu(gpu)
    if len(datasets_remaining) > 0:
        dataset = datasets_remaining.pop()
        gpu = gpu_manager.get_gpu()
        cmd = make_cmd(dataset, gpu, TASK)
        f = submit_job(cmd)
        f_to_gpu[f] = gpu
        # progress update
        if len(datasets_remaining) % 25 == 0:
            print('Progress: ', len(datasets_remaining), 'datasets remaining')
    else:  # no jobs remaining
        print('No jobs to submit.')


if __name__ == '__main__':
    datasets_base = os.path.join('data/motif/data/', TASK)
    datasets_remaining = get_datasets(datasets_base)
    print('Found %d datasets' % len(datasets_remaining))

    max_workers = GPUS * MAX_JOBS_PER_GPU
    pool = Pool(max_workers=max_workers)

    # start first jobs for all workers
    for _ in range(max_workers):
        dataset = datasets_remaining.pop()
        gpu = gpu_manager.get_gpu()
        cmd = make_cmd(dataset, gpu, TASK)
        f = submit_job(cmd)
        f_to_gpu[f] = gpu
