import torch
import gpytorch
import math
from gpytorch.mlls import PredictiveLogLikelihood, VariationalELBO
import sys 
sys.path.append("../")
from lolbo.utils.bo_utils.turbo import TurboState, update_state, generate_batch
from lolbo.utils.utils import (
    update_surr_model, 
    update_constraint_surr_models,
    update_models_end_to_end_with_constraints,
)
from lolbo.utils.bo_utils.ppgpr import GPModelDKL
import numpy as np
import os


class LOLBOState:

    def __init__(
        self,
        objective,
        train_x,
        train_y,
        train_z,
        train_c=None,
        k=1_000,
        minimize=False,
        num_update_epochs=2,
        init_n_epochs=20,
        learning_rte=0.01,
        bsz=10,
        acq_func='ei',
        verbose=True,
        iterations=0,
        spectral_norm=False,
        accumulated_z_next = [],
        accumulated_mean = [],
        accumulated_variance = [],
        accumulated_length = [],
        accumulated_y_next = [],

        
    ):
        self.objective          = objective         # objective with vae for particular task
        self.train_x            = train_x           # initial train x data
        self.train_y            = train_y           # initial train y data
        self.train_z            = train_z           # initial train z data
        self.train_c            = train_c           # initial constraint values data
        self.minimize           = minimize          # if True we want to minimize the objective, otherwise we assume we want to maximize the objective
        self.k                  = k                 # track and update on top k scoring points found
        self.num_update_epochs  = num_update_epochs # num epochs update models
        self.init_n_epochs      = init_n_epochs     # num epochs train surr model on initial data
        self.learning_rte       = learning_rte      # lr to use for model updates
        self.bsz                = bsz               # acquisition batch size
        self.acq_func           = acq_func          # acquisition function (Expected Improvement (ei) or Thompson Sampling (ts))
        self.verbose            = verbose
        self.iterations         = iterations        #iterations counter for saving gp mean, vars, x_next
        self.accumulated_z_next = accumulated_z_next
        self.accumulated_mean = accumulated_mean
        self.accumulated_variance = accumulated_variance
        self.accumulated_length =  accumulated_length
        self.accumulated_y_next = accumulated_y_next
        self.spectral_norm = spectral_norm
        assert acq_func in ["ei", "ts"]
        if minimize:
            self.train_y = self.train_y * -1
        self.ei_seen = 0
        self.progress_fails_since_last_e2e = 0
        self.tot_num_e2e_updates = 0
        # self.best_score_seen = torch.max(train_y)
        # self.best_x_seen = train_x[torch.argmax(train_y.squeeze())]
        self.initial_model_training_complete = False # initial training of surrogate model uses all data for more epochs
        self.new_best_found = False

        self.initialize_top_k()
        self.initialize_surrogate_model()
        self.initialize_tr_state()
        self.initialize_xs_to_scores_dict()


    def initialize_xs_to_scores_dict(self,):
        # put initial xs and ys in dict to be tracked by objective
        init_xs_to_scores_dict = {}
        for idx, x in enumerate(self.train_x):
            init_xs_to_scores_dict[x] = self.train_y.squeeze()[idx].item()
        self.objective.xs_to_scores_dict = init_xs_to_scores_dict


    def initialize_top_k(self):
        ''' Initialize top k x, y, and zs'''
        # if we have constriants, the top k are those that meet constraints!
        if self.train_c is not None: 
            bool_arr = torch.all(self.train_c <= 0, dim=-1) # all constraint values <= 0
            vaid_train_y = self.train_y[bool_arr]
            valid_train_z = self.train_z[bool_arr]
            valid_train_x = np.array(self.train_x)[bool_arr]
            valid_train_c = self.train_c[bool_arr] 
        else:
            vaid_train_y = self.train_y
            valid_train_z = self.train_z
            valid_train_x = self.train_x 

        if len(vaid_train_y) > 1:
            self.best_score_seen = torch.max(vaid_train_y)
            self.best_x_seen = valid_train_x[torch.argmax(vaid_train_y.squeeze())]

            # track top k scores found 
            self.top_k_scores, top_k_idxs = torch.topk(vaid_train_y.squeeze(), min(self.k, vaid_train_y.shape[0]))
            self.top_k_scores = self.top_k_scores.tolist() 
            top_k_idxs = top_k_idxs.tolist()
            self.top_k_xs = [valid_train_x[i] for i in top_k_idxs]
            self.top_k_zs = [valid_train_z[i].unsqueeze(-2) for i in top_k_idxs]
            if self.train_c is not None: 
                self.top_k_cs = [valid_train_c[i].unsqueeze(-2) for i in top_k_idxs]
        elif len(vaid_train_y) == 1:
            self.best_score_seen = vaid_train_y.item() 
            self.best_x_seen = valid_train_x.item() 
            self.top_k_scores = [self.best_score_seen]
            self.top_k_xs = [self.best_x_seen]
            self.top_k_zs = [valid_train_z] 
            if self.train_c is not None: 
                self.top_k_cs = [valid_train_c]
        else:
            print("No valid init data according to constraint(s)")
            self.best_score_seen = None
            self.best_x_seen = None 
            self.top_k_scores = []
            self.top_k_xs = []
            self.top_k_zs = []
            # add self .ei scores and graph
            if self.train_c is not None:
                self.top_k_cs = []


    def initialize_tr_state(self):
        if self.train_c is not None:  # if constrained 
            bool_arr = torch.all(self.train_c <= 0, dim=-1) # all constraint values <= 0
            vaid_train_y = self.train_y[bool_arr]
            valid_c_vals = self.train_c[bool_arr]
        else:
            vaid_train_y = self.train_y
            best_constraint_values = None
        
        if len(vaid_train_y) == 0:
            best_value = -torch.inf 
            if self.minimize:
                best_value = torch.inf
            if self.train_c is not None: 
                best_constraint_values = torch.ones(1,self.train_c.shape[1])*torch.inf
        else:
            best_value=torch.max(vaid_train_y).item()
            if self.train_c is not None: 
                best_constraint_values = valid_c_vals[torch.argmax(vaid_train_y)]
                if len(best_constraint_values.shape) == 1:
                    best_constraint_values = best_constraint_values.unsqueeze(-1) 
        # initialize turbo trust region state
        self.tr_state = TurboState( # initialize turbo state
            dim=self.train_z.shape[-1],
            batch_size=self.bsz, 
            best_value=best_value,
            best_constraint_values=best_constraint_values 
        )

        return self


    def initialize_constraint_surrogates(self):
        self.c_models = []
        self.c_mlls = []
        for i in range(self.train_c.shape[1]):
            likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
            n_pts = min(self.train_z.shape[0], 1024)
            c_model = GPModelDKL(self.train_z[:n_pts, :].cuda(), likelihood=likelihood,spectral_norm=self.spectral_norm).cuda()
            c_mll = PredictiveLogLikelihood(c_model.likelihood, c_model, num_data=self.train_z.size(-2))
            c_model = c_model.eval() 
            c_model = self.model.cuda() # changed this lets see what happens
            self.c_models.append(c_model)
            self.c_mlls.append(c_mll)
        return self  


    def initialize_surrogate_model(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
        n_pts = min(self.train_z.shape[0], 1024)
        self.model = GPModelDKL(self.train_z[:n_pts, :].cuda(), likelihood=likelihood,spectral_norm=self.spectral_norm).cuda()
        self.mll = PredictiveLogLikelihood(self.model.likelihood, self.model, num_data=self.train_z.size(-2))
        self.model = self.model.eval() 
        self.model = self.model.cuda()

        if self.train_c is not None:
            self.initialize_constraint_surrogates()

        return self


    def update_next(
        self,
        z_next_,
        y_next_,
        x_next_,
        c_next_=None,
        acquisition=False
    ):
        '''Add new points (z_next, y_next, x_next) to train data
            and update progress (top k scores found so far)
            and update trust region state
        '''
        
        if c_next_ is not None:
            if len(c_next_.shape) == 1:
                c_next_ = c_next_.unsqueeze(-1)
            valid_points = torch.all(c_next_ <= 0, dim=-1) # all constraint values <= 0
        else:
            valid_points = torch.tensor([True]*len(y_next_))
        z_next_ = z_next_.detach().cpu() 
        y_next_ = y_next_.detach().cpu()
        if len(y_next_.shape) > 1:
            y_next_ = y_next_.squeeze() 
        if len(z_next_.shape) == 1:
            z_next_ = z_next_.unsqueeze(0)
        progress = False
        for i, score in enumerate(y_next_):
            self.train_x.append(x_next_[i] )
            if valid_points[i]: # if y is valid according to constraints 
                if len(self.top_k_scores) < self.k: 
                    # if we don't yet have k top scores, add it to the list
                    self.top_k_scores.append(score.item())
                    self.top_k_xs.append(x_next_[i])
                    self.top_k_zs.append(z_next_[i].unsqueeze(-2))
                    if self.train_c is not None: # if constrained, update best constraints too
                        self.top_k_cs.append(c_next_[i].unsqueeze(-2))
                elif score.item() > min(self.top_k_scores) and (x_next_[i] not in self.top_k_xs):
                    # if the score is better than the worst score in the top k list, upate the list
                    min_score = min(self.top_k_scores)
                    min_idx = self.top_k_scores.index(min_score)
                    self.top_k_scores[min_idx] = score.item()
                    self.top_k_xs[min_idx] = x_next_[i]
                    self.top_k_zs[min_idx] = z_next_[i].unsqueeze(-2) # .to('cuda')
                    if self.train_c is not None: # if constrained, update best constraints too
                        self.top_k_cs[min_idx] = c_next_[i].unsqueeze(-2)
                #if this is the first valid example we've found, OR if we imporve 
                if (self.best_score_seen is None) or (score.item() > self.best_score_seen):
                    self.progress_fails_since_last_e2e = 0
                    progress = True
                    self.best_score_seen = score.item() #update best
                    self.best_x_seen = x_next_[i]
                    self.new_best_found = True
        if (not progress) and acquisition: # if no progress msde, increment progress fails
            self.progress_fails_since_last_e2e += 1
        y_next_ = y_next_.unsqueeze(-1)
        if acquisition:
            self.tr_state = update_state(
                state=self.tr_state,
                Y_next=y_next_,
                C_next=c_next_,
            )
        self.train_z = torch.cat((self.train_z, z_next_), dim=-2)
        self.train_y = torch.cat((self.train_y, y_next_), dim=-2)
        if c_next_ is not None:
            self.train_c = torch.cat((self.train_c, c_next_), dim=-2)

        return self


    def update_surrogate_model(self): 
        if not self.initial_model_training_complete:
            # first time training surr model --> train on all data
            n_epochs = self.init_n_epochs
            train_z = self.train_z
            train_y = self.train_y.squeeze(-1)
            train_c = self.train_c
        else:
            # otherwise, only train on most recent batch of data
            n_epochs = self.num_update_epochs
            train_z = self.train_z[-self.bsz:]
            train_y = self.train_y[-self.bsz:].squeeze(-1)
            if self.train_c is not None:
                train_c = self.train_c[-self.bsz:]
            else:
                train_c = None 
        print("Model parameters are on (update surr model):", next(self.model.parameters()).device)
        
        self.model = update_surr_model(
            self.model.cuda(),
            self.mll,
            self.learning_rte,
            train_z,
            train_y,
            n_epochs
        )
        if self.train_c is not None:
            self.c_models = update_constraint_surr_models(
                self.c_models,
                self.c_mlls,
                self.learning_rte,
                train_z,
                train_c,
                n_epochs
            )

        self.initial_model_training_complete = True

        return self


    def update_models_e2e(self):
        '''Finetune VAE end to end with surrogate model'''
        self.progress_fails_since_last_e2e = 0
        new_xs = self.train_x[-self.bsz:]
        new_ys = self.train_y[-self.bsz:].squeeze(-1).tolist()
        train_x = new_xs + self.top_k_xs
        train_y = torch.tensor(new_ys + self.top_k_scores).float()

        c_models = []
        c_mlls = []
        train_c = None 
        if self.train_c is not None:
            c_models = self.c_models 
            c_mlls = self.c_mlls
            new_cs = self.train_c[-self.bsz:] 
            # Note: self.top_k_cs is a list of (1, n_cons) tensors 
            if len(self.top_k_cs) > 0:
                top_k_cs_tensor = torch.cat(self.top_k_cs, -2).float() 
                train_c = torch.cat((new_cs, top_k_cs_tensor), -2).float() 
            else:
                train_c = new_cs 
            # train_c = torch.tensor(new_cs + self.top_k_cs).float() 
            
        print("Model parameters are on(e2e with constraints):", next(self.model.parameters()).device)
        self.objective, self.model = update_models_end_to_end_with_constraints(
            train_x=train_x,
            train_y_scores=train_y,
            objective=self.objective,
            model=self.model.cuda(),
            mll=self.mll,
            learning_rte=self.learning_rte,
            num_update_epochs=self.num_update_epochs,
            train_c_scores=train_c,
            c_models=c_models,
            c_mlls=c_mlls,
        )
        self.tot_num_e2e_updates += 1

        return self


    def recenter(self):
        '''Pass SELFIES strings back through
            VAE to find new locations in the
            new fine-tuned latent space
        '''

        self.objective.vae.eval()
        self.model.train()
        # which iteration did it recenter on. 

        optimize_list = [
            {'params': self.model.parameters(), 'lr': self.learning_rte} 
        ]
        if self.train_c is not None:
            for c_model in self.c_models:
                c_model.train() 
                optimize_list.append({f"params": c_model.parameters(), 'lr': self.learning_rte})
        optimizer1 = torch.optim.Adam(optimize_list, lr=self.learning_rte) 
        new_xs = self.train_x[-self.bsz:]
        train_x = new_xs + self.top_k_xs
        max_string_len = len(max(train_x, key=len))
        # max batch size smaller to avoid memory limit 
        #   with longer strings (more tokens) 
        bsz = max(1, int(2560/max_string_len))
        num_batches = math.ceil(len(train_x) / bsz) 
        for _ in range(self.num_update_epochs):
            for batch_ix in range(num_batches):
                optimizer1.zero_grad() 
                with torch.no_grad(): 
                    start_idx, stop_idx = batch_ix*bsz, (batch_ix+1)*bsz
                    batch_list = train_x[start_idx:stop_idx] 
                    z, _ = self.objective.vae_forward(batch_list)
                    out_dict = self.objective(z)
                    scores_arr = out_dict['scores'] 
                    constraints_tensor = out_dict['constr_vals']
                    valid_zs = out_dict['valid_zs']
                    xs_list = out_dict['decoded_xs']
                if len(scores_arr) > 0: # if some valid scores
                    scores_arr = torch.from_numpy(scores_arr)
                    if self.minimize:
                        scores_arr = scores_arr * -1
                    pred = self.model(valid_zs)
                    loss = -self.mll(pred, scores_arr.cuda())
                    if self.train_c is not None: 
                        for ix, c_model in enumerate(self.c_models):
                            pred2 = c_model(valid_zs.cuda())
                            loss += -self.c_mlls[ix](pred2, constraints_tensor[:,ix].cuda())
                    optimizer1.zero_grad()
                    loss.backward() 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer1.step() 
                    with torch.no_grad(): 
                        z = z.detach().cpu()
                        self.update_next(
                            z,
                            scores_arr,
                            xs_list,
                            c_next_=constraints_tensor
                        )
            torch.cuda.empty_cache()
        self.model.eval() 
        if self.train_c is not None:
            for c_model in self.c_models:
                c_model.eval() 

        return self


    def acquisition(self):
        '''Generate new candidate points, 
        evaluate them, and update data
        '''
        # 1. Generate a batch of candidates in 
        #   trust region using surrogate model
        if self.train_c is not None: # if constrained 
            constraint_model_list=self.c_models
        else:
            constraint_model_list = None 
        z_next, mean, variance, ei = generate_batch(
            state=self.tr_state,
            model=self.model,
            X=self.train_z,
            Y=self.train_y,
            batch_size=self.bsz, 
            acqf=self.acq_func,
            constraint_model_list=constraint_model_list,
        )
                  
        # 2. Evaluate the batch of candidates by calling oracle
        with torch.no_grad():
            out_dict = self.objective(z_next)
            z_next = out_dict['valid_zs']
            y_next = out_dict['scores']
            x_next = out_dict['decoded_xs']     
            c_next = out_dict['constr_vals']  
            if self.minimize:
                y_next = y_next * -1
                 
        # accumalte gp's, why does the shappe become heterogenous? 
        # fixing with deep copy's   
        self.accumulate_gp_predictions(z_next, mean, variance,self.tr_state,y_next)
        self.ei_seen = ei if ei is not None else -1e9
        self.iterations+=1
        if self.iterations % 10 == 0:
            self.save_gp_predictions_iteration()
        # 3. Add new evaluated points to dataset (update_next)
        if len(y_next) != 0:
            y_next = torch.from_numpy(y_next).float()
            self.update_next(
                z_next,
                y_next,
                x_next,
                c_next,
                acquisition=True
            )
        else:
            self.progress_fails_since_last_e2e += 1
            if self.verbose:
                print("GOT NO VALID Y_NEXT TO UPDATE DATA, RERUNNING ACQUISITOIN...")
    
    
    def accumulate_gp_predictions(self, z_next, mean, variance,tr_state,y_next):
        #helper function to append 
        # change when I get gpu 
        
        z_next, y_next = self.pre_process(z_next.cpu().numpy().copy(),y_next.copy())
        
        self.accumulated_z_next.append(z_next)
        self.accumulated_y_next.append(y_next) # already numpy object

        self.accumulated_mean.append(mean.cpu().numpy())
        self.accumulated_variance.append(variance.cpu().numpy())
        self.accumulated_length.append([tr_state.length] * len(mean))
        #is_recenter = [1 if self.iterations in self.recenter_history else 0] * len(mean)
        #self.accumulated_recenter_history.append(is_recenter)
        

    def pre_process(self,z_next,y_next):
        
        # target shapes should make the second dimeson more rigours 
        target_z_shape = (self.bsz,256)
        is_z_shape = target_z_shape == z_next.shape
        target_y_shape = (self.bsz,)
        is_y_shape = target_y_shape == y_next.shape
        print(f"Target z_next shape: {target_z_shape}, incoming z_next shape: {z_next.shape}")
        print(f"Target y_next shape: {target_y_shape}, incoming y_next shape: {y_next.shape}")
        if not is_z_shape:
            z_next = self.padding_helper(z_next,target_z_shape)
        if not is_y_shape:
            y_next = self.padding_helper(y_next,target_y_shape)
        
        
        return z_next,y_next
        
    def padding_helper(self,arr,target_shape):
        print(f"Padding arr with shape {arr.shape} to target shape {target_shape}")
        padded = np.full(target_shape,np.nan,dtype=arr.dtype)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
        padded[slices] = arr[slices]
        return padded


    def save_gp_predictions_iteration(self):
        # directory for saving data
        folder = f"gp_predictions/{self.objective.task_specific_args}"
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        # Save file with the current iteration number
        file_path = f"{folder}/gp_predictions_iter_{self.iterations // 10}.npz"
        for idx, item in enumerate(self.accumulated_z_next):
            print(f"Index: {idx}, Shape: {getattr(item, 'shape', 'N/A')}, Type: {type(item)}")
            
        np.savez(
                file_path,
                z_next=np.array(self.accumulated_z_next),
                y_next=np.array(self.accumulated_y_next),
                mean=np.array(self.accumulated_mean),
                variance=np.array(self.accumulated_variance),
                length=np.array(self.accumulated_length),
                #removed recenter
            )
        print(f"Saved accumulated data to {file_path}")
        self.accumulated_mean = []
        self.accumulated_variance = []
        self.accumulated_z_next = []
        self.accumulated_y_next = []
        self.accumulated_length = []
        self.accumulated_recenter_history = []