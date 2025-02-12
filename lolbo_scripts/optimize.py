import torch
import sys 
sys.path.append("../")
import random
import numpy as np 
import pandas as pd
import fire
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os
os.environ["WANDB_SILENT"] = "True"
from lolbo.lolbo import LOLBOState
from lolbo.latent_space_objective import LatentSpaceObjective
import signal 
import copy 
try:
    import wandb
    WANDB_IMPORTED_SUCCESSFULLY = True
except ModuleNotFoundError:
    WANDB_IMPORTED_SUCCESSFULLY = False


class Optimize(object):
    """
    Run LOLBO Optimization
    Args:
        task_id: String id for optimization task, by default the wandb project name will be f'optimize-{task_id}'
        seed: Random seed to be set. If None, no particular random seed is set
        track_with_wandb: if True, run progress will be tracked using Weights and Biases API
        wandb_entity: Username for your wandb account (valid username necessary iff track_with_wandb is True)
        wandb_project_name: Name of wandb project where results will be logged (if no name is specified, default project name will be f"optimimze-{self.task_id}")
        minimize: If True we want to minimize the objective, otherwise we assume we want to maximize the objective
        max_n_oracle_calls: Max number of oracle calls allowed (budget). Optimization run terminates when this budget is exceeded
        learning_rte: Learning rate for model updates
        acq_func: Acquisition function, must be either ei or ts (ei-->Expected Imporvement, ts-->Thompson Sampling)
        bsz: Acquisition batch size
        num_initialization_points: Number evaluated data points used to optimization initialize run
        init_n_update_epochs: Number of epochs to train the surrogate model for on initial data before optimization begins
        num_update_epochs: Number of epochs to update the model(s) for on each optimization step
        e2e_freq: Number of optimization steps before we update the models end to end (end to end update frequency)
        update_e2e: If True, we update the models end to end (we run LOLBO). If False, we never update end to end (we run TuRBO)
        k: We keep track of and update end to end on the top k points found during optimization
        verbose: If True, we print out updates such as best score found, number of oracle calls made, etc. 
    """
    def __init__(
        self,
        task_id: str,
        seed: int=None,
        track_with_wandb: bool=False,
        wandb_entity: str="",
        wandb_project_name: str="",
        minimize: bool=False,
        max_n_oracle_calls: int=200_000_000_000,
        learning_rte: float=0.001,
        acq_func: str="ei",
        bsz: int=10,
        num_initialization_points: int=100,
        init_n_update_epochs: int=80,
        num_update_epochs: int=2,
        e2e_freq: int=10,
        update_e2e: bool=True,
        k: int=1_000,
        verbose: bool=True,
        recenter_only=False,
        log_table_freq=10_000, 
        save_vae_ckpt=False,
    ):
        signal.signal(signal.SIGINT, self.handler)
        # add all local args to method args dict to be logged by wandb
        self.save_vae_ckpt = save_vae_ckpt
        self.log_table_freq = log_table_freq # log all collcted data every log_table_freq oracle calls 
        self.recenter_only = recenter_only # only recenter, no E2E 
        self.method_args = {}
        self.method_args['init'] = locals()
        del self.method_args['init']['self']
        self.seed = seed
        self.track_with_wandb = track_with_wandb
        self.wandb_entity = wandb_entity 
        self.task_id = task_id
        self.max_n_oracle_calls = max_n_oracle_calls
        self.verbose = verbose
        self.num_initialization_points = num_initialization_points
        self.e2e_freq = e2e_freq
        self.update_e2e = update_e2e
        self.set_seed()
        if wandb_project_name: # if project name specified
            self.wandb_project_name = wandb_project_name
        else: # otherwise use defualt
            self.wandb_project_name = f"optimimze-{self.task_id}"
        if not WANDB_IMPORTED_SUCCESSFULLY:
            assert not self.track_with_wandb, "Failed to import wandb, to track with wandb, try pip install wandb"
        if self.track_with_wandb:
            assert self.wandb_entity, "Must specify a valid wandb account username (wandb_entity) to run with wandb tracking"

        # initialize train data for particular task
        #   must define self.init_train_x, self.init_train_y, and self.init_train_z
        self.load_train_data()
        # initialize latent space objective (self.objective) for particular task
        self.initialize_objective()
        assert isinstance(self.objective, LatentSpaceObjective), "self.objective must be an instance of LatentSpaceObjective"
        assert type(self.init_train_x) is list, "load_train_data() must set self.init_train_x to a list of xs"
        if self.init_train_c is not None: # if constrained 
            assert torch.is_tensor(self.init_train_c), "load_train_data() must set self.init_train_c to a tensor of cs"
            assert self.init_train_c.shape[0] == len(self.init_train_x), f"load_train_data() must initialize exactly the same number of cs and xs, instead got {len(self.init_train_x)} xs and {self.init_train_c.shape[0]} cs"
        assert torch.is_tensor(self.init_train_y), "load_train_data() must set self.init_train_y to a tensor of ys"
        assert torch.is_tensor(self.init_train_z), "load_train_data() must set self.init_train_z to a tensor of zs"
        assert self.init_train_y.shape[0] == len(self.init_train_x), f"load_train_data() must initialize exactly the same number of ys and xs, instead got {self.init_train_y.shape[0]} ys and {len(self.init_train_x)} xs"
        assert self.init_train_z.shape[0] == len(self.init_train_x), f"load_train_data() must initialize exactly the same number of zs and xs, instead got {self.init_train_z.shape[0]} zs and {len(self.init_train_x)} xs"

        # initialize lolbo state
        self.lolbo_state = LOLBOState(
            objective=self.objective,
            train_x=self.init_train_x,
            train_y=self.init_train_y,
            train_z=self.init_train_z,
            train_c=self.init_train_c,
            minimize=minimize,
            k=k,
            num_update_epochs=num_update_epochs,
            init_n_epochs=init_n_update_epochs,
            learning_rte=learning_rte,
            bsz=bsz,
            acq_func=acq_func,
            verbose=verbose
        )


    def initialize_objective(self):
        ''' Initialize Objective for specific task
            must define self.objective object
            '''
        return self


    def load_train_data(self):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
            Must define the following:
                self.init_train_x (a list of x's)
                self.init_train_y (a tensor of scores/y's)
                self.init_train_c (a tensor of constraint values/c's)
                self.init_train_z (a tensor of corresponding latent space points)
        '''
        return self


    def set_seed(self):
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = False
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = False
        if self.seed is not None:
            torch.manual_seed(self.seed) 
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ["PYTHONHASHSEED"] = str(self.seed)

        return self


    def create_wandb_tracker(self):
        if self.track_with_wandb:
            config_dict = {k: v for method_dict in self.method_args.values() for k, v in method_dict.items()}
            self.tracker = wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                config=config_dict,
            ) 
            self.wandb_run_name = wandb.run.name
        else:
            self.tracker = None 
            self.wandb_run_name = 'no-wandb-tracking'
        
        return self


    def log_data_to_wandb_on_each_loop(self):
        if self.track_with_wandb: 
            dict_log = {
                "best_found":self.lolbo_state.best_score_seen,
                "n_oracle_calls":self.lolbo_state.objective.num_calls,
                "total_number_of_e2e_updates":self.lolbo_state.tot_num_e2e_updates,
                "best_input_seen":self.lolbo_state.best_x_seen,
                "ei_seen": self.lolbo_state.ei_seen,
                
            }
            dict_log[f"TR_length"] = self.lolbo_state.tr_state.length
            self.tracker.log(dict_log) 

        return self

    def run_lolbo(self): 
        ''' Main optimization loop
        '''
        # creates wandb tracker iff self.track_with_wandb == True
        self.create_wandb_tracker()
        last_logged_n_calls = 0 # log table + save vae ckpt every log_table_freq oracle calls
        #main optimization loop
        while self.lolbo_state.objective.num_calls < self.max_n_oracle_calls:
            self.log_data_to_wandb_on_each_loop()
            # update models end to end when we fail to make
            #   progress e2e_freq times in a row (e2e_freq=10 by default)
            if (self.lolbo_state.progress_fails_since_last_e2e >= self.e2e_freq) and self.update_e2e:
                if not self.recenter_only:
                    self.lolbo_state.update_models_e2e()
                self.lolbo_state.recenter()
                # Track this 
                if self.recenter_only:
                    self.lolbo_state.update_surrogate_model()
            else: # otherwise, just update the surrogate model on data
                self.lolbo_state.update_surrogate_model()
            # generate new candidate points, evaluate them, and update data
            self.lolbo_state.acquisition()
            if self.lolbo_state.tr_state.restart_triggered:
                self.lolbo_state.initialize_tr_state()
            # if a new best has been found, print out new best input and score:
            if self.lolbo_state.new_best_found:
                if self.verbose:
                    print("\nNew best found:")
                    self.print_progress_update()
                self.lolbo_state.new_best_found = False
            if (self.lolbo_state.objective.num_calls - last_logged_n_calls) >= self.log_table_freq:
                self.final_save = False 
                self.log_topk_table_wandb()
                last_logged_n_calls = self.lolbo_state.objective.num_calls


        # if verbose, print final results 
        if self.verbose:
            print("\nOptimization Run Finished, Final Results:")
            self.print_progress_update()

        # log top k scores and xs in table
        self.final_save = True 
        self.log_topk_table_wandb()
        self.tracker.finish()

        return self 


    def print_progress_update(self):
        ''' Important data printed each time a new
            best input is found, as well as at the end 
            of the optimization run
            (only used if self.verbose==True)
            More print statements can be added her as desired
        '''
        if self.track_with_wandb:
            print(f"Optimization Run: {self.wandb_project_name}, {wandb.run.name}")
        print(f"Best X Found: {self.lolbo_state.best_x_seen}")
        print(f"Best {self.objective.task_id} Score: {self.lolbo_state.best_score_seen}")
        print(f"Total Number of Oracle Calls (Function Evaluations): {self.lolbo_state.objective.num_calls}")

        return self
    

    def handler(self, signum, frame):
        # if we Ctrl-c, make sure we log top xs, scores found
        print("Ctrl-c hass been pressed, wait while we save all collected data...")
        self.final_save = True 
        self.log_topk_table_wandb()
        print("Now terminating wandb tracker...")
        self.tracker.finish() 
        msg = "Data now saved and tracker terminated, now exiting..."
        print(msg, end="", flush=True)
        exit(1)


    def log_topk_table_wandb(self):
        ''' After optimization finishes, log
            top k inputs and scores found
            during optimization '''
        if self.track_with_wandb and self.final_save:
            # save top k xs and ys 
            cols = ["Top K Scores", "Top K Strings"]
            data_list = []
            for ix, score in enumerate(self.lolbo_state.top_k_scores):
                data_list.append([ score, str(self.lolbo_state.top_k_xs[ix]) ])
            top_k_table = wandb.Table(columns=cols, data=data_list)
            self.tracker.log({f"top_k_table": top_k_table})

            # Additionally save table of ALL collected data 
            cols = ['All Scores', "All Strings"]
            data_list = []
            for ix, score in enumerate(self.lolbo_state.train_y.squeeze()):
                data_list.append([ score.item(), str(self.lolbo_state.train_x[ix]) ])
            try:
                full_table = wandb.Table(columns=cols, data=data_list)
                self.tracker.log({f"full_table": full_table})
            except:
                self.tracker.log({'save-data-table-failed':True})
            
        # We also want to save the fine-tuned VAE! 
        if self.save_vae_ckpt:
            n_calls = self.lolbo_state.objective.num_calls
            model = copy.deepcopy(self.lolbo_state.objective.vae)
            model = model.eval()
            model = model.cpu() 
            save_dir = 'finetuned_vae_ckpts/'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            model_save_path = save_dir + self.wandb_project_name + '_' + wandb.run.name + f'_finedtuned_vae_state_after_{n_calls}evals.pkl'  
            torch.save(model.state_dict(), model_save_path) 

        save_dir = 'optimization_all_collected_data/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        file_path = save_dir + self.wandb_project_name + '_' + wandb.run.name + '_all-data-collected.csv'
        df = {}
        df['train_x'] = np.array(self.lolbo_state.train_x)
        df['train_y'] = self.lolbo_state.train_y.squeeze().detach().cpu().numpy()  
        df = pd.DataFrame.from_dict(df)
        df.to_csv(file_path, index=None) 

        return self


    def done(self):
        return None


def new(**kwargs):
    return Optimize(**kwargs)

if __name__ == "__main__":
    fire.Fire(Optimize)
