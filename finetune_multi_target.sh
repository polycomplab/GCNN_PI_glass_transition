#!/bin/bash

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
config_name="config_finetune_2023_PI_combined_simple_subset_10K_regression_multitarget.py"
# config_name="config_finetune_two_targets.py"
# declare -a learning_rates=("1e-4" "5e-4" "1e-3" "5e-3" "1e-2")
# declare -a learning_rates=("1e-4" "5e-4" "1e-3" "5e-3")
# declare -a learning_rates=("1e-4" "6e-4" "1e-3" "1e-2")
# declare -a num_epochs=("30" "60" "100" "200" "500")
# declare -a num_epochs=("30" "60" "100")
# declare -a num_epochs=("30" "100")
# declare -a num_epochs=("60" "100" "300")
num_epochs="100"
learning_rate="1e-3"
# declare -a extra_loss_weight=("0.0" "0.01" "0.1" "0.5" "1.0")
# declare -a loss2_weights=("0.5" "1.0")
# declare -a num_task_layers=("2" "3")
# loss2_weight="0.5"

declare -a train_targets=("Tg"
                          "perm_He"
                          "perm_CH4"
                          "perm_CO2"
                          "perm_N2"
                          "perm_O2"
                          "perm_all"
                          "Tg_pretrained_on_perm_He"
                          "Tg_pretrained_on_perm_CH4"
                          "Tg_pretrained_on_perm_CO2"
                          "Tg_pretrained_on_perm_N2"
                          "Tg_pretrained_on_perm_O2"
                          "Tg_pretrained_on_perm_all"
                          "Tg_and_perm_He"
                          "Tg_and_perm_CH4"
                          "Tg_and_perm_CO2"
                          "Tg_and_perm_N2"
                          "Tg_and_perm_O2"
                          "perm_He_pretrained_on_Tg"
                          "perm_CH4_pretrained_on_Tg"
                          "perm_CO2_pretrained_on_Tg"
                          "perm_N2_pretrained_on_Tg"
                          "perm_O2_pretrained_on_Tg"
                          "perm_all_pretrained_on_Tg"
                          "Tg_and_perm_all"
                        )
# declare -a train_targets=(
#                           "Tg_pretrained_on_perm_He"
#                           "Tg_pretrained_on_perm_CH4"
#                           "Tg_pretrained_on_perm_CO2"
#                           "Tg_pretrained_on_perm_N2"
#                           "Tg_pretrained_on_perm_O2"
#                           "Tg_pretrained_on_perm_all"
#                           "perm_He_pretrained_on_Tg"
#                           "perm_CH4_pretrained_on_Tg"
#                           "perm_CO2_pretrained_on_Tg"
#                           "perm_N2_pretrained_on_Tg"
#                           "perm_O2_pretrained_on_Tg"
#                           "perm_all_pretrained_on_Tg"
#                         )


# declare -a train_targets=(
#                           "perm_CO2"
#                         )

declare -a devices=("6" "7" "8" "9")
declare -a pids=()

max_count=${#train_targets[@]}

counter=0  # counting started processes
i=0
train_tgts=${train_targets[counter]}
while :
do
    device_idx=${devices[i]}

    # MINWAIT=5
    # MAXWAIT=15
    # sleep $(( MINWAIT+RANDOM % (MAXWAIT-MINWAIT) )) &
    # secs="$((MINWAIT+RANDOM % (MAXWAIT-MINWAIT)))"
    # echo "$secs secs"

    # python3.7 test_gpus.py --device_idx="$device_idx" --seconds="$secs" &
    python3.7 main.py configs/"$config_name" --lr="$learning_rate" --epochs="$num_epochs" --device_idx="$device_idx" --train_targets="$train_tgts" 2>&1 | tee console_logs/$exp_name.txt &
    back_pid=$!
    # echo "started pid $back_pid on device $device_idx for $secs seconds"
    pids[$i]=$back_pid

    # start waiting only after all devices are loaded
    if (( ${#pids[@]}==${#devices[@]} )); then
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
    if [ "$counter" == "$max_count" ]; then
        # Waiting for the rest to finish"
        wait
        break
    fi
    train_tgts=${train_targets[counter]}
done


# for learning_rate in "${learning_rates[@]}"
# do
#     for e in "${num_epochs[@]}"
#     do
#         # for sub_size in "${subset_size[@]}"
#         for w in "${extra_loss_weight[@]}"
#         do
#             # for config_name in "${config_names[@]}"
#             # for pre_type in "${pre_types[@]}"
#             # for loss2_weight in "${loss2_weights[@]}"
#             # do
#             # for n_task_layers in "${num_task_layers[@]}"
#             # do
#                 # exp_name="$e"_epochs_lr_"$learning_rate"_loss2_weight_"$loss2_weight"_num_task_layers_"$n_task_layers"_"$config_name"
#             # exp_name="$e"_epochs_lr_"$learning_rate"_"$config_name"
#             # echo $exp_name
#             # exp_name=subset_size_"$sub_size"_"$e"_epochs_lr_"$learning_rate"_"$config_name"_pre_type_"$pre_type"
#             exp_name=extra_loss_weight_"$w"_"$e"_epochs_lr_"$learning_rate"_"$config_name"_pre_type_"$pre_type"
#             echo $exp_name

#             # TODO store pid with device in an array and wait on pids in the array to select a free device
#                 # python3 -c "import sys; print(sys.path)"
#                 # python3 --version
#             # python3.7 main.py configs/"$config_name" --lr="$learning_rate" --epochs="$e" --loss2_weight="$loss2_weight" --num_task_layers="$n_task_layers" --device_idx="$device_idx" 2>&1 | tee console_logs/$exp_name.txt &
#             # python3.7 main.py configs/"$config_name" --lr="$learning_rate" --epochs="$e" --device_idx="$device_idx" 2>&1 | tee console_logs/$exp_name.txt &
#             # python3.7 main.py configs/"$config_name" --subset_size="$sub_size" --lr="$learning_rate" --epochs="$e" --pre_type="$pre_type" --device_idx="$device_idx" 2>&1 | tee console_logs/$exp_name.txt &
#             sleep 3 &
#             # wait
#             if [ "$device_idx" == "6" ]; then
#                 device_idx="7"
#             elif [ "$device_idx" == "7" ]; then
#                 device_idx="8"
#             elif [ "$device_idx" == "8" ]; then
#                 device_idx="9"
#             elif [ "$device_idx" == "9" ]; then
#                 device_idx="6"
#                 wait
#             fi
#             # device_idx="7"
#             # done
#             # assumes 2 values of innermost loop, matching the number of devices
#             # done
#             # wait
#             # device_idx="6"
#         done
#         # wait
#     done
# done
