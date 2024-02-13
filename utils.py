import argparse
import configparser as ConfigParser
import ast
import numpy as np
#import ray

# ----------------------------------------------------------

def parse_args():
    """
        Parse the command line arguments
	"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-C','--config', default="myconfig.txt", required=True, help='Name of the input config file')
    parser.add_argument('-S','--seed', dest='random_state', default=42, type=int)
    
    args, __ = parser.parse_known_args()
    
    return vars(args)

# -----------------------------------------------------------

def parse_config(filename):
    
    config = ConfigParser.SafeConfigParser(allow_no_value=True)
    config.read(filename)
    
    # Build a nested dictionary with tasknames at the top level
    # and parameter values one level down.
    taskvals = dict()
    for section in config.sections():
        
        if section not in taskvals:
            taskvals[section] = dict()
        
        for option in config.options(section):
            # Evaluate to the right type()
            try:
                taskvals[section][option] = ast.literal_eval(config.get(section, option))
            except (ValueError,SyntaxError):
                err = "Cannot format field '{0}' in config file '{1}'".format(option,filename)
                err += ", which is currently set to {0}. Ensure strings are in 'quotes'.".format(config.get(section, option))
                raise ValueError(err)

    return taskvals, config

# -----------------------------------------------------------

