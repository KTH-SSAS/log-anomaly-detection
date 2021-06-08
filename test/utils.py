def set_args(bidir, model_type, token_level):
    """Prepares a dictionary of settings that can be used for testing."""
    # Common args (defaults, can be changed)
    args = {}

    args['bidirectional'] = bidir
    args['model_type'] = model_type
    args["trainer_config_file"] = 'config/config_trainer.json'
    if model_type == 'tiered-lstm':
        args['model_config_file'] = f'config/lanl_{token_level}_config_model_tiered.json'
    else:
        args['model_config_file'] = f'config/lanl_{token_level}_config_model.json'
    args['data_folder']= f'data/test_data/{token_level}_day_split'
    args['data_config_file'] = f'config/lanl_{token_level}_config_data.json'
    if token_level == 'word':
        args['jagged'] = False
    elif token_level == 'char':
        args['jagged'] = True
    else:
        raise RuntimeError("Unexpected token_level, args not prepared.")

    ### If model is both fwd and word tokenized
    if token_level == 'word' and not args['bidirectional']:
        args['skipsos'] = True
    else:
        args['skipsos'] = False
    
    # Return the prepared args
    return args