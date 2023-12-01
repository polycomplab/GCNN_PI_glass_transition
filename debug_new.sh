# A simple script to start 1-GPU pytorch training
# as soon as a GPU (from a list of available GPUs) becomes available.
# Intended for running a simple HPO on a grid.

declare -a devices=("6" "7" "8" "9")  # gpu indices
declare -a pids=()

max_count=20

counter=0  # counting started processes
i=0
while :
do
    device_idx=${devices[i]}

    MINWAIT=5
    MAXWAIT=15
    secs="$((MINWAIT+RANDOM % (MAXWAIT-MINWAIT)))"
    echo "starting a new process for $secs seconds on device $device_idx"

    # loads GPU memory for some time
    python3.7 test_gpus.py --device_idx="$device_idx" --seconds="$secs" &
    back_pid=$!
    pids[$i]=$back_pid

    # start waiting only after all devices are loaded
    if (( ${#pids[@]}==4 )); then
        wait -n

        # finding which device just finished and can be reused
        for pid_idx in "${!pids[@]}"
        do
            pid="${pids[$pid_idx]}"

            if ! (ps -p $pid > /dev/null)  # if just died
            then  # reuse gpu
                pids[pid_idx]=-1
                i=$pid_idx
                break
            fi
        done
    else
        i=$(($i + 1))
        i=$(($i % ${#devices[@]}))
    fi
    
    counter=$(($counter+1))
    if (( "$counter" == "$max_count" )); then
        # Waiting for the rest to finish
        wait
        break
    fi
done
