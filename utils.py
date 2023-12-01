import torch


def load_pretrained_weights(model, config):
    state_dict = torch.load(
            config.finetune.pretrained_weights,
            map_location=torch.device(config.device_index))
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model_state_dict = model.state_dict()
    for name, tensor in model_state_dict.items():
        # assert name in state_dict
        if name not in state_dict:
            print(f'weights for layer {name} cannot be loaded because'\
                    ' the layer didn\'t exist in that checkpoint. It will be initialized randomly.')
            continue
        if tensor.shape != state_dict[name].shape:
            print(f'weights for layer {name} will not be loaded because of mismatching shape:')
            print(f'weights shape: {state_dict[name].shape}, required shape: {tensor.shape}')
            state_dict[name] = tensor
    for name in state_dict:
        if name not in model_state_dict:
            print(f'checkpoint has weights for missing layer {name}. It won\'t be loaded')
    return state_dict
