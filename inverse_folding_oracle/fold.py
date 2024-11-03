from transformers import AutoTokenizer, EsmForProteinFolding
import torch 
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import uuid
import os 
import esm 
from esm.inverse_folding.util import CoordBatchConverter
# device = "cuda:0" if torch.cuda.is_available() else "cpu" 
if not torch.cuda.is_available():
    print("NO GPU AVAILABLE")
    assert 0 

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def inverse_fold(target_pdb_id, chain_id="A", model=None):
    '''This function is used to convert a pdb file to a sequence using inverse folding
    Default to chain A, since we only support folding a single sequence at the moment
    '''
    # Crystal: 
    pdb_path = f"../inverse_folding_oracle/target_cif_files/{target_pdb_id}.cif" 
    if model is None:
        model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    structure = esm.inverse_folding.util.load_structure(pdb_path, chain_id)
    coords, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)
    # Lower sampling temperature typically results in higher sequence recovery but less diversity
    sampled_seq = model.sample(coords, temperature=1e-6)
    return sampled_seq 


def inverse_fold_many_seqs(target_pdb_id, num_seqs, chain_id="A", model=None):
    '''This function is used to convert a pdb file to a sequence using inverse folding
    Default to chain A, since we only support folding a single sequence at the moment
    returns a list of inverse folded seqs
    '''
    # Crystal: 
    pdb_path = f"../inverse_folding_oracle/target_cif_files/{target_pdb_id}.cif" 
    if model is None:
        model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    structure = esm.inverse_folding.util.load_structure(pdb_path, chain_id)
    coords, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)
    # Lower sampling temperature typically results in higher sequence recovery but less diversity
    sampled_seqs = model.sample(coords, temperature=1, num_seqs=num_seqs) 
    return sampled_seqs 

def seq_to_pdb(seq, save_path="./output.pdb", model=None):
    # This function is used to fold a sequence to a pdb file
    # Load the model and tokenizer
    if model is None:
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

    model = model.to('cpu') 
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    ### Added by Natalie, remove tokens in UNIREF that are unsupported by ESM  
    seq = seq.replace("-", "") 
    seq = seq.replace("U", "") 
    seq = seq.replace("X", "") 
    seq = seq.replace("Z", "") 
    seq = seq.replace("O", "") 
    seq = seq.replace("B", "")

    tokenized_input = tokenizer([seq], return_tensors="pt", add_special_tokens=False)['input_ids'].to('cpu') 

    with torch.no_grad():
        output = model(tokenized_input)

    output = convert_outputs_to_pdb(output)
    with open(save_path, "w") as f:
        # the convert_outputs_to_pdb function returns a list of pdb files, since we only have one sequence, we only need the first one
        f.write(output[0])
    return

def fold_aa_seq(aa_seq, esm_model=None):
    if esm_model is None:
        esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to('cpu') 
    if not os.path.exists("temp_pdbs/"):
        os.mkdir("temp_pdbs/")
    folded_pdb_path = f"temp_pdbs/{uuid.uuid1()}.pdb"
    seq_to_pdb(seq=aa_seq, save_path=folded_pdb_path, model=esm_model)
    return folded_pdb_path 

def load_esm_if_model():
    if_model, if_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    if_model = if_model.eval() 
    if_model = if_model.to('cpu') 
    return if_model, if_alphabet 

def get_gvp_encoding(pdb_path, chain_id='A', model=None, alphabet=None):
    with torch.no_grad():
        # This function is used to get the GVP encoding of a sequence
        if model is None:
            model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval()
        model = model.to('cpu') 

        structure = esm.inverse_folding.util.load_structure(pdb_path, chain_id)

        # Extracting Coordinates from Structure
        coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
        coords = torch.tensor(coords) # .to('cpu') 

        batch_converter = CoordBatchConverter(alphabet) # .to('cpu') 
        batch = [(coords, None, native_seq)]

        coords, confidence, strs, tokens, padding_mask = batch_converter(batch)
        confidence = confidence.to('cpu') 
        gvp_out = model.encoder.forward_embedding(coords.to('cpu'), padding_mask=padding_mask.to('cpu'), confidence=confidence)[1]['gvp_out']

    # gvp_out.shape   torch.Size([1, 123, 512]) 
    return gvp_out


def aa_seq_to_gvp_encoding(aa_seq, if_model=None, if_alphabet=None, fold_model=None):
    if (if_model is None) or (if_alphabet is None):
        if_model, if_alphabet = load_esm_if_model()
    if fold_model is None: 
        fold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to('cpu') 
    folded_pdb = fold_aa_seq(aa_seq, esm_model=fold_model)
    with torch.no_grad():
        encoding = get_gvp_encoding(pdb_path=folded_pdb, model=if_model, alphabet=if_alphabet) 
    return encoding


def aa_seqs_list_to_avg_gvp_encodings(aa_seq_list, if_model=None, if_alphabet=None, fold_model=None):
    with torch.no_grad():
        if (if_model is None) or (if_alphabet is None):
            if_model, if_alphabet = load_esm_if_model()
        if fold_model is None: 
            fold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to('cpu') 
        folded_pdbs = [fold_aa_seq(aa_seq, esm_model=fold_model) for aa_seq in aa_seq_list]

        # V1 get individually  
        # encodings = [get_gvp_encoding(pdb_path=folded_pdb, model=if_model, alphabet=if_alphabet) for folded_pdb in folded_pdbs]
        # avg_encodings = [encoding.nanmean(-2) for encoding in encodings]
        # avg_encodings = torch.cat(avg_encodings, 0) 
        
        # Faster version: 
        encodings = get_gvp_encoding_batch(
            pdb_path=folded_pdbs, 
            chain_id='A', 
            model=if_model, 
            alphabet=if_alphabet, 
            save_memory=False # set to true to forward through GVP on cpu 
        )  
        avg_encodings = encodings.nanmean(-2)
    return avg_encodings 



def get_gvp_encoding_batch(pdb_path=[], chain_id='A', model=None, alphabet=None, save_memory=True):
    with torch.no_grad():
        if save_memory:
            device = "cpu"
        else:
            device = "cuda:0"
        # This function is used to get the GVP encoding of a sequence
        if model is None:
            model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval()
        model = model.to(device)

        batch = []

        for pdb in pdb_path:
            structure = esm.inverse_folding.util.load_structure(pdb, chain_id)

            # Extracting Coordinates from Structure
            coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
            coords = torch.tensor(coords).to(device)

            batch.append((coords, None, native_seq))

        batch_converter = CoordBatchConverter(alphabet)

        coords_batch, confidence_batch, strs, tokens, padding_mask_batch = batch_converter(batch, device=device)
        confidence_batch = confidence_batch.to(device)

        gvp_out = model.encoder.forward_embedding(coords_batch, padding_mask=padding_mask_batch, confidence=confidence_batch)[1]['gvp_out']

    return gvp_out


if __name__ == "__main__":
    aa_seq = "AABBCCDDEEFFGG"
    folded_pdb = fold_aa_seq(aa_seq, esm_model=None) 
    import pdb 
    pdb.set_trace() 



