import torch
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

def calc_cost(weights, path):

	mask = path > 0
	cost = torch.sum(weights * mask , dim=1)

	return cost

# -----------------------------------------------------------

def exact_match_accuracy(true_paths, suggested_paths):

	accuracy = torch.all(torch.eq(true_paths, suggested_paths),  dim=1).to(torch.float32).mean()

	return accuracy

# -----------------------------------------------------------

def exact_cost_accuracy(true_paths, suggested_paths, weights):

	eps = 1e-6
	true_costs = calc_cost(weights, true_paths)
	pred_costs = calc_cost(weights, suggested_paths)
	
	accuracy = (torch.abs(true_costs - pred_costs) < eps).to(torch.float32).mean()

	return accuracy