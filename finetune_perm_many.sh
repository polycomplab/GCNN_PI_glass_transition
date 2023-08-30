#!/bin/sh

# declare -a config_names=("config_finetune_perm_100K.py"
declare -a config_names=("config_finetune_perm_full.py"
                         "config_finetune_perm_not_pretrained.py"
                         )
declare -a learning_rates=("5e-5" "7e-5" "1e-4" "3e-4" "5e-4" "7e-4" "1e-3" "3e-3" "5e-3" "7e-3" "1e-2")
declare -a num_epochs=("5" "10" "20" "30" "40" "50" "60" "70" "80" "90" "100" "120" "150" "200" "250" "300" "400" "500" "1000")
device_idx="6"

for learning_rate in "${learning_rates[@]}"
do
    for e in "${num_epochs[@]}"
    do
        for config_name in "${config_names[@]}"
        do
            exp_name="$e""_epochs_lr_""$learning_rate""_""$config_name"
            echo $exp_name

        # TODO store pid with device in an array and wait on pids in the array to select a free device
            # python3 -c "import sys; print(sys.path)"
            # python3 --version
            python3.7 main.py configs/"$config_name" --lr="$learning_rate" --epochs="$e" --device_idx="$device_idx" 2>&1 | tee console_logs/$exp_name.txt &
            device_idx="7"
        done
        # assumes 2 configs
        wait
        device_idx="6"
    done
done
