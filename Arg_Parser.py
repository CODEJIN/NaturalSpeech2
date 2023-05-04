from argparse import Namespace

def Recursive_Parse(args_dict):
    parsed_dict = {}
    for key, value in args_dict.items():
        if isinstance(value, dict):
            value = Recursive_Parse(value)
        parsed_dict[key]= value

    args = Namespace()
    args.__dict__ = parsed_dict
    return args

def To_Non_Recursive_Dict(
    args: Namespace
    ):
    parsed_dict = {}
    for key, value in args.__dict__.items():
        if isinstance(value, Namespace):
            value_dict = To_Non_Recursive_Dict(value)
            for sub_key, sub_value in value_dict.items():
                parsed_dict[f'{key}.{sub_key}'] = sub_value
        else:
            parsed_dict[key] = value

    return parsed_dict