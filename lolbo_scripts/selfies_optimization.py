import sys
sys.path.append("../")
import fire
from lolbo_scripts.optimize import Optimize
from lolbo.info_transformer_vae_objective import InfoTransformerVAEObjective
import math 
import pandas as pd 
import torch 
import math 
from lolbo.SelfiesObjective import SelfiesObjective
from lolbo.utils.mol_utils.load_data import load_molecule_train_data, compute_train_zs


class SelfiesOptimization(Optimize):
    """
    Run LOLBO Optimization with GuacaMole 
    Args:
        path_to_vae_statedict: Path to state dict of pretrained VAE,
        max_string_length: Limit on string length that can be generated by VAE 
            (without a limit we can run into OOM issues)
        dim: dimensionality of latent space of VAE
        constraint1_min_threshold: min allowed value for constraint 1 (None --> unconstrained)
        constraint2_max_threshold: max allowed value for constraint 2 (None --> unconstrained)
    """
    def __init__(
        self,
        path_to_vae_statedict: str="/users/8/joh22439/home/my_lolbo_cleaner/lolbo/utils/mol_utils/selfies_vae/state_dict/SELFIES-VAE-state-dict.pt",
        dim: int=1024,
        task_specific_args: str='logp', #  of additional arg to be passed into objective funcion 
        max_string_length: int=1024,
        constraint_function_ids: list=[], # list of strings identifying the black box constraint function to use
        constraint_thresholds: list=[], # list of corresponding threshold values (floats)
        constraint_types: list=[], # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
        init_data_path: str="../initialization_data/guacamol_train_data_first_20k.csv",
        **kwargs,
    ):
        self.path_to_vae_statedict = path_to_vae_statedict
        self.dim = dim 
        self.max_string_length = max_string_length
        self.task_specific_args = task_specific_args 
        self.init_data_path = init_data_path
        # To specify constraints, pass in 
        #   1. constraint_function_ids: a list of constraint function ids, 
        #   2. constraint_thresholds: a list of thresholds, 
        #   3. constraint_types: a list of threshold types (must be "min" or "max" for each)
        # Empty lists indicate that the problem is unconstrained, 1 or more constraints can be added 
        assert len(constraint_function_ids) == len(constraint_thresholds)
        assert len(constraint_thresholds) == len(constraint_types)
        self.constraint_function_ids = constraint_function_ids # list of strings identifying the black box constraint function to use
        self.constraint_thresholds = constraint_thresholds # list of corresponding threshold values (floats)
        self.constraint_types = constraint_types # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
        
        super().__init__(**kwargs,task_specific_args=self.task_specific_args) 

        # add args to method args dict to be logged by wandb 
        self.method_args['opt1'] = locals()
        del self.method_args['opt1']['self']


    def initialize_objective(self):
        # initialize objective 
        self.objective = SelfiesObjective(
            task_id=self.task_id, # string id for your task
            task_specific_args=self.task_specific_args, # of additional args to be passed into objective funcion 
            path_to_vae_statedict=self.path_to_vae_statedict, # state dict for VAE to load in
            max_string_length=self.max_string_length, # max string length that VAE can generate
            dim=self.dim, # dimension of latent search space
            constraint_function_ids=self.constraint_function_ids, # list of strings identifying the black box constraint function to use
            constraint_thresholds=self.constraint_thresholds, # list of corresponding threshold values (floats)
            constraint_types=self.constraint_types, # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
        )
        # if train zs have not been pre-computed for particular vae, compute them 
        #   by passing initialization selfies through vae 
        
        self.init_train_z = compute_train_zs(
                self.objective,
                self.init_train_x,
            )
        # compute initial constriant values
        self.init_train_c = self.objective.compute_constraints(self.init_train_x)

        return self


    def load_train_data(self):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
            Must define the following:
                self.init_train_x (a list of x's)
                self.init_train_y (a tensor of scores/y's)
                self.init_train_z (a tensor of corresponding latent space points)
            '''
        assert self.num_initialization_points <= 20_000 
        smiles, selfies, zs, ys = load_molecule_train_data(
            # the specicic arg
            task_id=self.task_specific_args,
            num_initialization_points=self.num_initialization_points,
            path_to_vae_statedict=self.path_to_vae_statedict
        )
        self.init_train_x, self.init_train_z, self.init_train_y = smiles, zs, ys
        if self.verbose:
            print("Loaded initial training data")
            print("train y shape:", self.init_train_y.shape)
            print(f"train x list length: {len(self.init_train_x)}\n")

        # create initial smiles to selfies dict
        self.init_smiles_to_selfies = {}
        for ix, smile in enumerate(self.init_train_x):
            self.init_smiles_to_selfies[smile] = selfies[ix]

        return self 


if __name__ == "__main__":
    fire.Fire(SelfiesOptimization)
