# run_all.py
import multiprocessing
import subprocess
import os

scripts = [
    "people_detect.py",
    "rf2.py"
]

def run_script(script_path):
    full_path = os.path.join(os.path.dirname(__file__), script_path)
    subprocess.run(['python', full_path])

if __name__ == "__main__":
    processes = []

    for script in scripts:
        p = multiprocessing.Process(target = run_script, args = (script,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()