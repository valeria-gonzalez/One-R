def print_to_file(mode, dict_args):
    """Print a given dictionary to a file.

    Args:
        mode (char): mode to open the file ('w' or 'a')
        dict_args (dict): dictionary with data description and data
        
    Returns:
        None
    """
    
    with open('output.txt', mode) as file:
        for key, value in dict_args.items():
            file.write(f"{key}: \n\n")
            file.write(f"{value}\n\n")