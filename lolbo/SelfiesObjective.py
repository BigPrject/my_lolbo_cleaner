import numpy as np
import torch 
import selfies as sf 
from lolbo.utils.mol_utils.selfies_vae.model_positional_unbounded import SELFIESDataset, InfoTransformerVAE
from lolbo.utils.mol_utils.selfies_vae.data import collate_fn
from lolbo.latent_space_objective import LatentSpaceObjective
from lolbo.utils.mol_utils.mol_utils import GUACAMOL_TASK_NAMES
from your_tasks.your_objective_functions import OBJECTIVE_FUNCTIONS_DICT


import pkg_resources
# make sure molecule software versions are correct: 
assert pkg_resources.get_distribution("selfies").version == '2.0.0'
assert pkg_resources.get_distribution("rdkit-pypi").version == '2022.3.1'
assert pkg_resources.get_distribution("molsets").version == '0.3.1'


class SelfiesObjective(LatentSpaceObjective):
    '''Selfies class supports all molecule optimization
        tasks and uses the SELFIES VAE by default '''

    def __init__(
        self,
        task_id='guacamol',
        task_specific_args='pdop',
        path_to_vae_statedict="../lolbo/utils/mol_utils/selfies_vae/state_dict/SELFIES-VAE-state-dict.pt",
        xs_to_scores_dict={},
        max_string_length=1024,
        num_calls=0,
        smiles_to_selfies={},
        constraint_function_ids=[], # list of strings identifying the black box constraint function to use
        constraint_thresholds=[], # list of corresponding threshold values (floats)
        constraint_types=[], # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
        dim = 256
    ):
        # we only want guacamol tasks right now
        assert task_specific_args in GUACAMOL_TASK_NAMES + ["logp"]
        print("task_specific_args:", task_specific_args)
        self.task_specific_args = task_specific_args
        self.objective_function = OBJECTIVE_FUNCTIONS_DICT[task_id](self.task_specific_args)


        self.dim                    = 256 # SELFIES VAE DEFAULT LATENT SPACE DIM
        self.path_to_vae_statedict  = path_to_vae_statedict # path to trained vae stat dict
        self.max_string_length      = max_string_length # max string length that VAE can generate
        self.smiles_to_selfies      = smiles_to_selfies # dict to hold computed mappings form smiles to selfies strings
        self.constraint_functions       = []
        super().__init__(
            num_calls=num_calls,
            xs_to_scores_dict=xs_to_scores_dict,
            task_id=task_id,
        )
        

    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        z = z.to('cpu')
        self.vae = self.vae.eval()
        self.vae = self.vae.to('cpu') 
        # sample molecular string form VAE decoder
        sample = self.vae.sample(z=z.reshape(-1, 2, 128))
        # grab decoded selfies strings
        decoded_selfies = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]
        # decode selfies strings to smiles strings (SMILES is needed format for oracle)
        decoded_smiles = []
        for selfie in decoded_selfies:
            smile = sf.decoder(selfie)
            decoded_smiles.append(smile)
            # save smile to selfie mapping to map back later if needed
            self.smiles_to_selfies[smile] = selfie

        return decoded_smiles


    def query_oracle(self, x):
        ''' Input: 
                a  list  input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        # method assums x is a list already
        score = self.objective_function(x)

        return score


    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        self.dataobj = SELFIESDataset()
        self.vae = InfoTransformerVAE(dataset=self.dataobj)
        # load in state dict of trained model:
        if self.path_to_vae_statedict:
            state_dict = torch.load(self.path_to_vae_statedict, map_location=torch.device('cpu'))
            self.vae.load_state_dict(state_dict, strict=True)
        self.vae = self.vae.to('cpu')
        self.vae = self.vae.eval()
        # set max string length that VAE can generate
        self.vae.max_string_length = self.max_string_length


    def vae_forward(self, xs_batch):
        ''' Input: 
                a list xs 
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae 
                    (ie reconstruction error)
        '''
        # assumes xs_batch is a batch of smiles strings 
        X_list = []
        for smile in xs_batch:
            try:
                # avoid re-computing mapping from smiles to selfies to save time
                selfie = self.smiles_to_selfies[smile]
            except:
                selfie = sf.encoder(smile)
                self.smiles_to_selfies[smile] = selfie
            tokenized_selfie = self.dataobj.tokenize_selfies([selfie])[0]
            encoded_selfie = self.dataobj.encode(tokenized_selfie).unsqueeze(0)
            X_list.append(encoded_selfie)
        X = collate_fn(X_list)
        dict = self.vae(X.to('cpu'))
        vae_loss, z = dict['loss'], dict['z']
        z = z.reshape(-1,self.dim)

        return z, vae_loss


if __name__ == "__main__":
    # testing molecule objective
    obj1 = MoleculeObjective(task_id='pdop' ) 
    print(obj1.num_calls)
    dict1 = obj1(torch.randn(10,256))
    print(dict1['scores'], obj1.num_calls)
    dict1 = obj1(torch.randn(3,256))
    print(dict1['scores'], obj1.num_calls)
    dict1 = obj1(torch.randn(1,256))
    print(dict1['scores'], obj1.num_calls)
