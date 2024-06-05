import subprocess
import multiprocessing

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        with open("error.txt", "a") as file:
            file.write(f"Error running command: {e}")

if __name__ == "__main__":
    with open("error.txt", "w") as file:
        file.write("Error File")
    total = 5
    commands= []
    for i in range(total):
        command = f'python /workspace/CS762_Project/Model/generate_new_examples_gpt.py --part {i+1} --total {total} 2>&1 | tee output_part_{i+1}.log'
        commands.append(command)

    # Create a multiprocessing pool with the desired number of processes
    num_processes = len(commands)  # You can adjust this as needed
    pool = multiprocessing.Pool(processes=num_processes)

    # Execute the commands in parallel
    pool.map(run_command, commands)

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()
