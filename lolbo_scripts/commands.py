import subprocess

commands = [
"python3 selfies_optimization.py --task_id guacamol --task_specific_args med1 --track_with_wandb True --wandb_entity joh22439-university-of-minnesota --num_initialization_points 1000  --max_n_oracle_calls 10000 --bsz 10 --dim 1024 --max_string_length 400 - run_lolbo - done",
"python3 selfies_optimization.py --task_id guacamol --task_specific_args med2 --track_with_wandb True --wandb_entity joh22439-university-of-minnesota --num_initialization_points 1000  --max_n_oracle_calls 10000 --bsz 10 --dim 1024 --max_string_length 400 - run_lolbo - done",
"python3 selfies_optimization.py --task_id guacamol --task_specific_args pdop --track_with_wandb True --wandb_entity joh22439-university-of-minnesota --num_initialization_points 1000  --max_n_oracle_calls 10000 --bsz 10 --dim 1024 --max_string_length 400 - run_lolbo - done"

]
print("Running Tasks")

for command in commands:
    try:
        task = command.split(' ')[5]
        
        print(f"Running:{task} task, Base model no spectral norm with pll")
        subprocess.run(command, shell=True) 
    except Exception as e:
        print(f"{task} failed due to {e}")