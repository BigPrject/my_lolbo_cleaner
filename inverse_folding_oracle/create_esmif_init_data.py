

# quick instructions on how to sample in batches:
# instead of installing esm from their repo, install from my fork of it at 
# pip install git+https://github.com/Yimeng-Zeng/esm.git
# and instead of doing sampled_seq = model.sample(coords, temperature=1, device=device)
# we now do sampled_seq = model.sample_batch(coords, temperature=1, device=device, num_samples=20)
# which returns a list of sequences

import sys 
sys.path.append("../")
import esm 
import os 
import math 
from inverse_folding_oracle.aa_seq_to_tm_score import aa_seq_to_tm_score
from transformers import EsmForProteinFolding
import pandas as pd 
import argparse 

def create_esmif_init_data(
    target_pdb_id, 
    total_num_seqs_generate=100,
    batch_size=20, 
    save_data_path=None, 
    chain_id="A",
    batch_mode=True,
):
    if save_data_path is None:
        save_data_path = f"../initialization_data/esmif_target_structure_{target_pdb_id}.csv"
    target_pdb_path = f"target_pdb_files/target_structure_{target_pdb_id}.pdb"
    assert os.path.exists(target_pdb_path)

    if_model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    if_model = if_model.to('cpu') 
    if_model = if_model.eval()
    structure = esm.inverse_folding.util.load_structure(target_pdb_path, chain_id)
    coords, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)

    n_iters = math.ceil(total_num_seqs_generate/batch_size) 
    all_sampled_seqs = []
    for _ in range(n_iters):
        if batch_mode:
            sampled_seqs = if_model.sample(coords, temperature=1, num_seqs=batch_size, device="cuda:0") 
        else:
            sampled_seqs = []
            for _ in range(batch_size):
                seq1 = if_model.sample(coords, temperature=1, device="cuda:0") 
                sampled_seqs.append(seq1) 

        all_sampled_seqs = all_sampled_seqs + sampled_seqs
    
    if_model = if_model.cpu() 
    del(if_model)

    esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    esm_model = esm_model.eval() 
    esm_model = esm_model.to('cpu')
    
    scores_list = [] 
    for seq in all_sampled_seqs:
        score = aa_seq_to_tm_score(
            aa_seq=seq, 
            target_pdb_path=target_pdb_path,
            esm_model=esm_model,
        ) 
        scores_list.append(score)
    
    # Now save data 
    df = {}
    df["x"] = all_sampled_seqs
    df["y"] = scores_list 
    df = pd.DataFrame.from_dict(df)
    df.to_csv(save_data_path, index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--target_pdb_id', type=int, default=0 ) 
    parser.add_argument('--bsz', type=int, default=20 ) 
    parser.add_argument('--total_num_seqs_generate', type=int, default=1_000 )
    parser.add_argument('--save_data_path', default=None )
    parser.add_argument('--batch_mode', type=bool, default=True ) 
    args = parser.parse_args() 

    create_esmif_init_data(
        target_pdb_id=args.target_pdb_id, 
        total_num_seqs_generate=args.total_num_seqs_generate,
        batch_size=args.bsz, 
        save_data_path=args.save_data_path, 
        batch_mode=args.batch_mode,
    )

# python3 create_esmif_init_data.py --target_pdb_id 0 --bsz 20 --total_num_seqs_generate 1000 



