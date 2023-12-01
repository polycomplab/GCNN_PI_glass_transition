#!/bin/sh

# declare -a config_names=("config_finetune_perm_100K.py"
# declare -a config_names=("config_finetune_perm_full.py"
#                          "config_finetune_perm_not_pretrained.py"
# declare -a config_names=("config_finetune_one_target.py"
#                          "config_finetune_one_target_pretrained_perm.py"
#                          )
# declare -a pre_types=("5000_just_trained_v2"
#                       "100K_just_trained"
#                       "100K_just_trained_v2"
#                       )
pre_type="None"
# config_name="config_finetune_2023_PI_exp_last.py"
config_name="config_finetune_2023_PI_exp_last_simple_subsets.py"
# config_name="config_finetune_two_targets.py"
# declare -a learning_rates=("1e-4" "5e-4" "1e-3" "5e-3" "1e-2")
# declare -a learning_rates=("1e-4" "5e-4" "1e-3" "5e-3")
declare -a learning_rates=("1e-4" "6e-4" "1e-3" "1e-2")
# declare -a num_epochs=("30" "60" "100" "200" "500")
# declare -a num_epochs=("30" "60" "100")
# declare -a num_epochs=("30" "100")
declare -a num_epochs=("10" "100" "300")
declare -a subset_size=("250" "500" "750")
# declare -a loss2_weights=("0.5" "1.0")
# declare -a num_task_layers=("2" "3")
# loss2_weight="0.5"
device_idx="6"

for learning_rate in "${learning_rates[@]}"
do
    for e in "${num_epochs[@]}"
    do
        for sub_size in "${subset_size[@]}"
        do
            # for config_name in "${config_names[@]}"
            # for pre_type in "${pre_types[@]}"
            # for loss2_weight in "${loss2_weights[@]}"
            # do
            # for n_task_layers in "${num_task_layers[@]}"
            # do
                # exp_name="$e"_epochs_lr_"$learning_rate"_loss2_weight_"$loss2_weight"_num_task_layers_"$n_task_layers"_"$config_name"
            # exp_name="$e"_epochs_lr_"$learning_rate"_"$config_name"
            # echo $exp_name
            exp_name=subset_size_"$sub_size"_"$e"_epochs_lr_"$learning_rate"_"$config_name"_pre_type_"$pre_type"
            echo $exp_name

            # TODO store pid with device in an array and wait on pids in the array to select a free device
                # python3 -c "import sys; print(sys.path)"
                # python3 --version
            # python3.7 main.py configs/"$config_name" --lr="$learning_rate" --epochs="$e" --loss2_weight="$loss2_weight" --num_task_layers="$n_task_layers" --device_idx="$device_idx" 2>&1 | tee console_logs/$exp_name.txt &
            # python3.7 main.py configs/"$config_name" --lr="$learning_rate" --epochs="$e" --device_idx="$device_idx" 2>&1 | tee console_logs/$exp_name.txt &
            python3.7 main.py configs/"$config_name" --subset_size="$sub_size" --lr="$learning_rate" --epochs="$e" --pre_type="$pre_type" --device_idx="$device_idx" 2>&1 | tee console_logs/$exp_name.txt &
            # wait
            if [ "$device_idx" == "6" ]; then
                device_idx="7"
            elif [ "$device_idx" == "7" ]; then
                device_idx="8"
            elif [ "$device_idx" == "8" ]; then
                device_idx="6"
            fi
            # device_idx="7"
            # done
            # assumes 2 values of innermost loop, matching the number of devices
            # done
            # wait
            # device_idx="6"
        done
        wait
    done
done
