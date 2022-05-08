import numpy as np
import mdtraj as md
from enspara.geometry.pockets import xyz_to_mdtraj
import enspara.geometry.libdist

def get_nearby_residues(holo_xtal, apo_xtal, ligand_resn, dist_cutoff, computeshared = False):

    #handle bricks
    if ligand_resn == "None":
        print("no valid ligands found")
        return [] #prot_info, np.array([], int), np.array([], int), np.array([], int), np.array([], int), np.zeros(apo_xtal.top.n_residues, int)]

    #handle everything else
    if type(ligand_resn) == str: #select single ligands
        ligand_select_str = f"resname '{ligand_resn}'"
    elif len(ligand_resn)>1:     #select multiple ligands
        #assemble query for all ligands listed
        #ligand_select_str = f"resname '{ligand_resn[0]}'"
        molecule_queries = []
        for lname in ligand_resn:
            molecule_queries.append(f"resname '{lname}'")
            #ligand_select_str += f" or resname '{lname}'"
        ligand_select_str = " or ".join(molecule_queries)
    else:                        #use hardcoded query
        ligand_select_str = ligand_resn[0]

    #if len(molecule_queries) == 0:
    #    print("no valid ligands found")
    #    return False

    #get indices of ligand and protein atoms
    ligand_ai = holo_xtal.top.select(f"({ligand_select_str}) and not element H")
    protein_ai = holo_xtal.top.select("protein and not element H")

    #get indices of ligand-coordinating protein atoms; index 0 indicates the 0th trajectory frame
    lining_ai = md.compute_neighbors(holo_xtal, dist_cutoff, ligand_ai, haystack_indices=protein_ai, periodic = False)[0]

    #get residue indices and numbers
    #lining_resids = np.unique([holo_xtal.top.atom(i).residue.index for i in lining_ai])    #0-indices of the lining residues in the mdtraj structure
    #lining_resseqs = np.unique([holo_xtal.top.atom(i).residue.resSeq for i in lining_ai])  #rcsb pdb residue numbers of the lining residues

    #select backbones and sidechains containing the ligand-coordinating atoms

    #obtain all backbone and sidechain atom indices
    sidechain_iis = holo_xtal.top.select("sidechain")
    bb_iis = holo_xtal.top.select("backbone")

    sele = []
    for i in lining_ai:

        #get pdb residue number
        resi = holo_xtal.top.atom(i).residue.resSeq

        #separate sidechains and backbones
        if i in sidechain_iis:
            sele.append("sidechain and resSeq %s" % str(resi))
        elif i in bb_iis:
            sele.append("backbone and resSeq %s" % str(resi))
        else:
            print(f'error: atom {i} is in neither sidechain nor backbone')
            break

    sele = np.unique(sele)

    prot_iis_holo = np.concatenate([holo_xtal.top.select(f"{sel} and not element H") for sel in sele]).ravel()
    prot_iis_apo = np.concatenate([apo_xtal.top.select(f"{sel} and not element H") for sel in sele]).ravel()

    #compute atom indices of residues present in both apo and holo structures; usually for RMSD calculations
    if computeshared:
        #note that this code does not introduce any duplicate indices and hence no application of np.unique is needed
        prot_iis_holo_shared = []
        prot_iis_apo_shared = []

        for sel in sele:
            #include only indices of atoms in residues present in apo and holo structures
            if len(holo_xtal.top.select(f"{sel} and not element H")) > 0 and len(apo_xtal.top.select(f"{sel} and not element H")) > 0:
                prot_iis_holo_shared.append(holo_xtal.top.select(f"{sel} and not element H"))
                prot_iis_apo_shared.append(apo_xtal.top.select(f"{sel} and not element H"))

        prot_iis_holo_shared = np.concatenate(prot_iis_holo).ravel()
        prot_iis_apo_shared = np.concatenate(prot_iis_apo).ravel()

        return [sele, prot_iis_holo, prot_iis_apo, prot_iis_holo_shared, prot_iis_apo_shared]

    else:
        return [sele, prot_iis_holo, prot_iis_apo]

#-------------------------------------------------------------------------------

#calculate cryptic pocket volume using heavy atoms
def cryptic_pocket_vol(xtal, pv_pdb, pocket_ai, savepock=False, savename = "", output_dir = ""):

    if savepock:
        #coordinates of the ligsite pocket elements owned by atoms of ligand-coordinating residue segments
        pocket_ele = [] # for qc

    protein_iis = xtal.top.select("protein and not element H") #protein atom indices

    #the number of pocket elements which are nearer to each ligand-lining atom than to any other atom
    # = the number of pocket elements owned by that atom
    num_owned_pock_ele = np.zeros(len(pocket_ai))

    for pocket_coord in pv_pdb.xyz[0]: #loop over pocket coordinates

        #calculate the distance to all crystal structure atoms and sort to find the closest atoms
        dists = enspara.geometry.libdist.euclidean(xtal.xyz[0], pocket_coord)
        atom_distance_iis = np.argsort(dists)

        #go down the list of nearest atoms to find the nearest protein atom to the pocket
        for i in atom_distance_iis:
            if i in protein_iis:
                if i in pocket_ai:

                    #Keep track of which atom in the cryptic pocket lining owns how many pocket elements
                    #and the coordinates of the pocket elements filling the cryptic pocket.

                    #get the index of the protein atom in the list of ligand-coordinating residue atoms
                    #and add 1 to the number of pocket atoms that it owns
                    num_owned_pock_ele[np.where(pocket_ai==i)[0][0]] += 1

                    if savepock:
                        pocket_ele.append(pocket_coord) #for qc

                    break #don't assign the pocket coordinate to multiple owners

                else:
                    #a protein atom outside of the pocket lining owns this pocket element
                    break
            else:
                #do not assign pocket volume to hydrogens, ligand atoms, or other non-protein atoms such as structural ions
                continue

    if savepock:
        #save a pdb of holo pocket elements owned by ligand coordinating residues for QC
        pocket_pdb = xyz_to_mdtraj(np.array(pocket_ele))
        pocket_pdb.save(f"{output_dir}/qc-output/{savename}-lig-coord-resi-adjacent-ligsite-pockets.pdb")

    return num_owned_pock_ele.sum()

#-------------------------------------------------------------------------------
#get the paths to the cluster centers and pockets for a given FAST gen
#[zip, number of centers]

def paths_zipped(gen, n_gens, apo_id, fast_path, ctr="*"):

    if gen != n_gens-1:
        path_strings = [f"{fast_path}/FASTPockets-{apo_id}/msm/old/centers_masses{gen}/state{ctr}-00.pdb",
                        f"{fast_path}/FASTPockets-{apo_id}/msm/old/pocket_analysis{gen}/state{ctr}/state{ctr}_pockets.pdb"]
    else:
        path_strings = [f"{fast_path}/FASTPockets-{apo_id}/msm/centers_masses/state{ctr}-00.pdb",
                        f"{fast_path}/FASTPockets-{apo_id}/msm/pocket_analysis/state{ctr}/state{ctr}_pockets.pdb"]

    ctrs_fns = glob.glob(path_strings[0])

    return [zip(np.sort(ctrs_fns), np.sort(glob.glob(path_strings[1]))), len(ctrs_fns)]

#-------------------------------------------------------------------------------

def fast_paths(fast_path, apo_id, gen, n_gens, ctr):

    ctr = str(ctr).zfill(6)

    if gen != n_gens-1:
        path_strings = [f"{fast_path}/FASTPockets-{apo_id}/msm/old/centers_masses{gen}/state{ctr}-00.pdb",
                        f"{fast_path}/FASTPockets-{apo_id}/msm/old/pocket_analysis{gen}/state{ctr}/state{ctr}_pockets.pdb"]
    else:
        path_strings = [f"{fast_path}/FASTPockets-{apo_id}/msm/centers_masses/state{ctr}-00.pdb",
                        f"{fast_path}/FASTPockets-{apo_id}/msm/pocket_analysis/state{ctr}/state{ctr}_pockets.pdb"]

    return path_strings

#-------------------------------------------------------------------------------

#calculate the cryptic pocket volumes, RMSDs, and SASAs for all cluster centers in the given gen
def vols_by_gen(gen, n_gens, apo_id, prot_masses_pocket_ai, fast_path): #, holo_pocket_coord):

    #get zipped filepaths to cluster centers and pockets and the number of cluster centers
    #[zip, number of centers]
    filepaths_zipped = paths_zipped(gen, n_gens, apo_id, fast_path)

    #store calculated properties
    pvols = []
    #rmsds = []

    count = 0 #for print output

    #calculate properties each cluster center
    for ctr_fn, pocket_fn in filepaths_zipped[0]:

        ctr = md.load(ctr_fn)

        #rmsds.append(calc_rmsd(ctr, holo_pocket_coord, prot_masses_pocket_res_iis)[0])

        #load pockets if applicable and calculate volume owned by them
        try:
            pocket = md.load(pocket_fn)
            pvols.append(cryptic_pocket_vol(ctr, pocket, prot_masses_pocket_res_iis))
        except: #some structures have zero pocket volume, whereupon ligsite crashes
            #and no pocket file is generated, causing md.load to crash
            print("pocket skipped") #or an error occured in cryptic_pocket_vol
            pvols.append(0)
            continue

        #for print output of calculation progress
        count+=1
        if count % 50 == 0:
            print(count / filepaths_zipped[1])

    return pvols #[pvols, rmsds]
