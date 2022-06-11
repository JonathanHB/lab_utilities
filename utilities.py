import numpy as np
import glob

import mdtraj as md
from enspara.geometry import pockets
from enspara.geometry.pockets import xyz_to_mdtraj
import enspara.geometry.libdist
import scipy
from scipy.spatial import distance

import time

def get_nearby_residues(holo_xtal, apo_xtal, ligand_resn, dist_cutoff, computeshared = False, shift = 'f'):

    #handle bricks
    if ligand_resn == "None":
        print("no valid ligands found")
        return []

    #handle everything else
    if type(ligand_resn) == str: #select single ligands
        ligand_select_str = f"resname '{ligand_resn}'"

    elif len(ligand_resn)>1:     #select multiple ligands
        #assemble query for all ligands listed
        molecule_queries = []
        for lname in ligand_resn:
            molecule_queries.append(f"resname '{lname}'")
        ligand_select_str = " or ".join(molecule_queries)

    else:                        #use hardcoded query
        ligand_select_str = ligand_resn[0]

    #get indices of ligand and protein atoms
    ligand_ai = holo_xtal.top.select(f"({ligand_select_str}) and not element H")
    protein_ai = holo_xtal.top.select("protein and not element H")

    #get indices of ligand-coordinating protein atoms; index 0 indicates the 0th trajectory frame
    lining_ai = md.compute_neighbors(holo_xtal, dist_cutoff, ligand_ai, haystack_indices=protein_ai, periodic = False)[0]

    #get residue indices and numbers for the holo structure
    lining_resids = np.unique([holo_xtal.top.atom(i).residue.index for i in lining_ai])    #0-indices of the lining residues in the mdtraj structure
    lining_resseqs = np.unique([holo_xtal.top.atom(i).residue.resSeq for i in lining_ai])  #rcsb pdb residue numbers of the lining residues

    #select backbones and sidechains containing the ligand-coordinating atoms

    #obtain all backbone and sidechain atom indices
    sidechain_iis = holo_xtal.top.select("sidechain")
    bb_iis = holo_xtal.top.select("backbone")

    sele = []
    for i in lining_ai:

        #get pdb residue number
        resi = holo_xtal.top.atom(i).residue.resSeq
        #print(holo_xtal.top.atom(i).residue)

        #separate sidechains and backbones
        if i in sidechain_iis:
            sele.append("sidechain and resSeq %s" % str(resi))
        elif i in bb_iis:
            sele.append("backbone and resSeq %s" % str(resi))
        else:
            print(f'error: atom {i} is in neither sidechain nor backbone')
            break

    #remove redundant entries
    sele = np.unique(sele)

    #get apo and holo atom indices for ligand-lining residue segments
    prot_iis_holo = np.concatenate([holo_xtal.top.select(f"{sel} and not element H") for sel in sele]).ravel()
    prot_iis_apo = np.concatenate([apo_xtal.top.select(f"{sel} and not element H") for sel in sele]).ravel()

    #-------------------------------------------optional additional outputs; currently mutually exclusive---------------------------------------------

    #if a number, normally 0, has been entered for the shift argument,
    # calculate all matching resolved atoms in ligand-lining residues
    if shift != 'f':
        # all atom indices from residues shared between apo and holo and resolved to the same degree in each
        prot_iis_holo_matching = []
        prot_iis_apo_matching = []

        # loop through all residues (whole residues not sidechains/backbones)
        for lrseq in lining_resseqs:
            holo_atoms = [[ai, holo_xtal.top.atom(ai)] for ai in holo_xtal.top.select(f"resSeq {lrseq} and protein and not element H")]
            apo_atoms = {str(apo_xtal.top.atom(ai)).split('-')[-1]:ai for ai in apo_xtal.top.select(f"resSeq {lrseq + shift} and protein and not element H")}

            #print(holo_atoms)
            #print(apo_atoms)
            #print("-----------------------------------")

            #find all the atoms which match between apo and holo to return; should be robust against atom order
            # variation and arbitrary missing atom permutations
            for hatom in holo_atoms:
                try:
                    prot_iis_apo_matching.append(apo_atoms[str(hatom[1]).split('-')[-1]]) #this line must go first because if it crashes the following line should not run
                    prot_iis_holo_matching.append(hatom[0])

                except:
                    print(f"{hatom[1]} not found in apo")

            #print(prot_iis_holo_matching)
            #print(prot_iis_apo_matching)

        return [sele, prot_iis_holo, prot_iis_apo, lining_resids, lining_resseqs, prot_iis_holo_matching, prot_iis_apo_matching]

    #---------------------------------------------------------------------------------------------------------------------

    #compute atom indices of ligand-lining residue segments present in both apo and holo structures; usually for RMSD calculations
    if computeshared:
        #note that this code does not introduce any duplicate indices and hence no application of np.unique is needed

        #all atom indices from residues shared between apo and holo, even if those residues are incomplete
        prot_iis_holo_shared = []
        prot_iis_apo_shared = []

        for sel in sele:
            #include only indices of atoms in residues present in apo and holo structures
            if len(holo_xtal.top.select(f"{sel} and not element H")) == len(apo_xtal.top.select(f"{sel} and not element H")): # > 0:
                prot_iis_holo_shared.append(holo_xtal.top.select(f"{sel} and not element H"))
                prot_iis_apo_shared.append(apo_xtal.top.select(f"{sel} and not element H"))

        prot_iis_holo_shared = np.concatenate(prot_iis_holo).ravel()
        prot_iis_apo_shared = np.concatenate(prot_iis_apo).ravel()

        return [sele, prot_iis_holo, prot_iis_apo, prot_iis_holo_shared, prot_iis_apo_shared]


    return [sele, prot_iis_holo, prot_iis_apo, lining_resids, lining_resseqs]

#-------------------------------------------------------------------------------

# get a list of matching apo and holo c-alpha indices
def get_all_matching_ca(holo_xtal, apo_xtal, shift=0):
    holo_rseqs = [next(r.atoms).residue.resSeq for r in holo_xtal.top.residues]
    apo_rseqs = [next(r.atoms).residue.resSeq - shift for r in apo_xtal.top.residues]
    shared_rseqs = set(holo_rseqs).intersection(set(apo_rseqs))

    holo_ai = []
    apo_ai = []

    for rs in shared_rseqs:
        if rs >= 0:

            holo_ca = holo_xtal.top.select(f"resSeq {rs} and protein and name CA and element C")
            apo_ca = apo_xtal.top.select(f"resSeq {rs + shift} and protein and name CA and element C")

            if len(holo_ca) == 1 and len(apo_ca) == 1:
                holo_ai.append(holo_ca[0])
                apo_ai.append(apo_ca[0])

            else:
                print(f"skipping residue {rs}; apo Calpha selection has {len(apo_ca)} atoms and holo Calpha selection has {len(holo_ca)} atoms")

    return [holo_ai, apo_ai]


#-------------------------------------------------------------------------------

def get_nearby_residues_2chains(holo_xtal, apo_xtal, ligand_resn, holo_dist_cutoff, apo_dist_cutoff, shift):

    #handle bricks
    if ligand_resn == "None":
        print("no valid ligands found")
        return []

    #handle everything else
    if type(ligand_resn) == str: #select single ligands
        ligand_select_str = f"resname '{ligand_resn}'"

    elif len(ligand_resn)>1:     #select multiple ligands
        #assemble query for all ligands listed
        molecule_queries = []
        for lname in ligand_resn:
            molecule_queries.append(f"resname '{lname}'")
        ligand_select_str = " or ".join(molecule_queries)

    else:                        #use hardcoded query
        ligand_select_str = ligand_resn[0]

    #get indices of ligand and protein atoms
    ligand_ai = holo_xtal.top.select(f"({ligand_select_str}) and not element H")
    protein_ai_holo = holo_xtal.top.select("protein and not element H")
    protein_ai_apo = holo_xtal.top.select("protein and not element H")


    #get matching alpha carbons to align apo and holo structures and align them
    ca_matching = get_all_matching_ca(holo_xtal, apo_xtal, shift)
    apo_xtal.superpose(holo_xtal, atom_indices = ca_matching[1], ref_atom_indices = ca_matching[0]) #modifies apo_xtal in place

    #get indices of ligand-coordinating protein atoms; index 0 indicates the 0th trajectory frame
    #holo
    lining_ai_holo = md.compute_neighbors(holo_xtal, holo_dist_cutoff, ligand_ai, haystack_indices=protein_ai_holo, periodic = False)[0]
    lining_resseqs_holo = np.unique([holo_xtal.top.atom(i).residue.resSeq for i in lining_ai_holo])  #rcsb pdb residue numbers of the lining residues
    print(type(lining_ai_holo))

    #apo (trickier since the ligand and protein are in different structures so md.compute_neighbors can't be used)
    apo_xyz = apo_xtal.atom_slice(protein_ai_apo).xyz[0]
    ligand_xyz = holo_xtal.atom_slice(ligand_ai).xyz[0]

    apo_lig_dists = scipy.spatial.distance.cdist(apo_xyz, ligand_xyz)
    print(apo_lig_dists.shape)
    print(apo_lig_dists.min(axis = 1).shape)

    lining_ai_apo = np.where(apo_lig_dists.min(axis = 1) < apo_dist_cutoff)
    print(type(lining_ai_apo[0]))
    #lining_ai_apo = md.compute_neighbors(holo_xtal, apo_dist_cutoff, ligand_ai, haystack_indices=protein_ai, periodic = False)[0]

    lining_resseqs_apo = np.unique([apo_xtal.top.atom(i).residue.resSeq + shift for i in list(lining_ai_apo)])  #rcsb pdb residue numbers of the lining residues

    #combine apo and holo resseqs
    lining_resseqs = lining_resseqs_apo+lining_resseqs_holo

    # calculate all matching resolved atoms in ligand-lining residues
    # all atom indices from residues shared between apo and holo and resolved to the same degree in each
    prot_iis_holo_matching = []
    prot_iis_apo_matching = []

    # loop through all residues (whole residues not sidechains/backbones)
    for lrseq in lining_resseqs:
        holo_atoms = [[ai, holo_xtal.top.atom(ai)] for ai in holo_xtal.top.select(f"resSeq {lrseq} and protein and not element H")]
        apo_atoms = {str(apo_xtal.top.atom(ai)).split('-')[-1]:ai for ai in apo_xtal.top.select(f"resSeq {lrseq + shift} and protein and not element H")}

        #print(holo_atoms)
        #print(apo_atoms)
        #print("-----------------------------------")

        #find all the atoms which match between apo and holo to return; should be robust against atom order
        # variation and arbitrary missing atom permutations
        for hatom in holo_atoms:
            try:
                prot_iis_apo_matching.append(apo_atoms[str(hatom[1]).split('-')[-1]]) #this line must go first because if it crashes the following line should not run
                prot_iis_holo_matching.append(hatom[0])

            except:
                print(f"{hatom[1]} not found in apo")

        #print(prot_iis_holo_matching)
        #print(prot_iis_apo_matching)

    return [lining_resseqs, prot_iis_holo_matching, prot_iis_apo_matching]

    # get residue indices and numbers for the holo structure
    # lining_resids_holo = np.unique([holo_xtal.top.atom(i).residue.index for i in lining_ai])    #0-indices of the lining residues in the mdtraj structure


#-------------------------------------------------------------------------------

#calculate cryptic pocket volume using heavy atoms
def cryptic_pocket_vol(xtal, pv_pdb, pocket_ai, savepock=False, savename = "", output_dir = ""):

    if savepock:
        #coordinates of the ligsite pocket elements owned by atoms of ligand-coordinating residue segments
        #used for qc and figures
        pocket_ele = []

    #all protein atom indices
    #note that no speedup could be obtained here by the use of list comprehension instead of mdtraj
    protein_iis = xtal.top.select("protein and not element H")

    #the number of pocket elements which are nearer to each ligand-lining atom than to any other atom
    # = the number of pocket elements owned by that atom
    #this may be done with a dictionary instead, which is more intuitive and eliminates the np.where() statement below, but there appears to be no speed advantage to this
    num_owned_pock_ele = np.zeros(len(pocket_ai))

    #loop over pocket coordinates to assign each one to a protein atom
    for pocket_coord in pv_pdb.xyz[0]:

        #calculate the distance from the current pocket element to all
        #protein atoms and sort to find the closest protein atom
        ##dists = enspara.geometry.libdist.euclidean(xtal.xyz[0][protein_iis], pocket_coord)
        ##atom_distance_iis = np.argsort(dists)

        #assign the grid point to the nearest protein atom
        ##i = protein_iis[atom_distance_iis[0]]
        i = protein_iis[np.argsort(enspara.geometry.libdist.euclidean(xtal.xyz[0][protein_iis], pocket_coord))[0]]

        if i in pocket_ai:
            #Keep track of which atom in the cryptic pocket lining owns how many pocket elements
            #and the coordinates of the pocket elements filling the cryptic pocket.

            #get the index of the protein atom in the list of ligand-coordinating residue atoms
            #and add 1 to the number of pocket atoms that it owns
            num_owned_pock_ele[np.where(pocket_ai==i)[0][0]] += 1

            if savepock:
                pocket_ele.append(pocket_coord)

    if savepock:
        #save a pdb of holo pocket elements owned by ligand coordinating residues for QC

        pocket_pdb = xyz_to_mdtraj(np.array(pocket_ele))
        pocket_pdb.save(f"{output_dir}/qc-output/{savename}-lig-coord-resi-adjacent-ligsite-pockets.pdb")

    return num_owned_pock_ele.sum()


#-------------------------------------------------------------------------------
#get the paths to the cluster centers and pockets for a given FAST gen
#[zip, number of centers]

def paths_zipped(fast_path, apo_id, gen, n_gens, tem1=False):

    fast_upper = f"{fast_path}/FASTPockets-{apo_id}/msm"

    if tem1:
        fast_upper = f"{fast_path}/FAST-pockets/msm"

    if gen != n_gens-1:
        path_strings = [f"{fast_upper}/old/centers_masses{gen}/state*-00.pdb",
                        f"{fast_upper}/old/pocket_analysis{gen}/state*/state*_pockets.pdb"]
    else:
        path_strings = [f"{fast_upper}/centers_masses/state*-00.pdb",
                        f"{fast_upper}/pocket_analysis/state*/state*_pockets.pdb"]

    ctrs_fns = np.sort(glob.glob(path_strings[0]))
    pocket_fns = np.sort(glob.glob(path_strings[1]))

    if len(ctrs_fns) == len(pocket_fns):
        return [zip(ctrs_fns, pocket_fns), len(ctrs_fns)]
    else:
        return [zip(ctrs_fns, ctrs_fns), len(ctrs_fns)]


#-------------------------------------------------------------------------------
#get the paths to the cluster centers and pockets for a given center in a given FAST gen
#[path to cluster center structure, path to cluster center pockets]

def fast_paths_1ctr(fast_path, apo_id, gen, n_gens, ctr):

    ctr = str(ctr).zfill(6)
    fast_upper = f"{fast_path}/FASTPockets-{apo_id}/msm"

    if gen != n_gens-1:
        path_strings = [f"{fast_upper}/old/centers_masses{gen}/state{ctr}-00.pdb",
                        f"{fast_upper}/old/pocket_analysis{gen}/state{ctr}/state{ctr}_pockets.pdb"]
    else:
        path_strings = [f"{fast_upper}/centers_masses/state{ctr}-00.pdb",
                        f"{fast_upper}/pocket_analysis/state{ctr}/state{ctr}_pockets.pdb"]

    return path_strings

#-------------------------------------------------------------------------------




#-------------------------------------------------------------------------------

#calculate the cryptic pocket volumes, RMSDs, and SASAs for all cluster centers in the given gen
def vols_by_gen(gen, n_gens, apo_id, prot_masses_pocket_ai, fast_path, calc_pockets = False, ligsiteparams=[], tem1_in=False):

    #get zipped filepaths to cluster centers and pockets and the number of cluster centers
    #[zip, number of centers]
    filepaths_zipped = paths_zipped(fast_path, apo_id, gen, n_gens, tem1=tem1_in)

    #store calculated properties
    pvols = []

    count = 0 #for print output

    #calculate properties each cluster center
    for ctr_fn, pocket_fn in filepaths_zipped[0]:
        ctr = md.load(ctr_fn)

        #load pockets if applicable and calculate volume owned by them
        try:
            if not calc_pockets:
                pocket = md.load(pocket_fn)
            else:
                pocket = pockets.get_pockets(ctr, ligsiteparams[0], ligsiteparams[1], ligsiteparams[2], ligsiteparams[3])[0]

            pvols.append(cryptic_pocket_vol(ctr, pocket, prot_masses_pocket_ai))
        #some structures have zero pocket volume, whereupon ligsite crashes
        #and no pocket file is generated, causing md.load to crash
        except:
            print("pocket skipped") #or an error occured in cryptic_pocket_vol
            pvols.append(0)

        #for print output of calculation progress
        count+=1
        if count % 50 == 0:
            print(count / filepaths_zipped[1])

    return pvols

