# Description:   Extracts dendritic E/I synapse densities (#synapses/um) for each neuron within a given circuit target in isolation
# Author:        C. Pokorny
# Created:       10/02/2021
# Last modified: 27/04/2021

import sys
import time
from joblib import Parallel, delayed
from datetime import datetime
import progressbar

import numpy as np
import pandas as pd

from bluepy import Circuit
from bluepy import Cell
from bluepy import Synapse
import neurom as nm
from neurom.core.types import NeuriteType

import warnings
from bluepysnap.exceptions import BluepySnapDeprecationWarning

# Creates data splits [for parallel processing]
def create_data_splits(circuit_config, circuit_target, N_split):
    # Load circuit
    circuit = Circuit(circuit_config)
    cell_ids = circuit.cells.ids(circuit_target)
    cell_ids_split = np.split(cell_ids, np.cumsum([np.ceil(len(cell_ids) / N_split).astype(int)] * (N_split - 1)))
    
    print(f'INFO: Created {N_split} data splits for {circuit_config} using target "{circuit_target}" with {len(cell_ids)} cells', flush=True)
    
    return cell_ids_split


# Create table with cell positions and dentritic synapse density for given set of GIDs [for easy parallelization]
def create_cell_table(circuit_config, circuit_target, gids):
    
    # Disable deprecation/user warnings
    warnings.filterwarnings('ignore', category=BluepySnapDeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Load circuit
    circuit = Circuit(circuit_config)
    pre_cell_ids = circuit.cells.ids(circuit_target)
    num_cells = len(gids)
    proj_names = list(circuit.config['projections'].keys())
    
    print(f'INFO: Creating cell table (incl. {len(proj_names)} projections) for {num_cells} cells: {gids[:3]}..{gids[-3:]}', flush=True)
    
    # Extract cell positions
    cell_table = circuit.cells.get(gids, properties=[Cell.X, Cell.Y, Cell.Z])
    
    # Get dendrite length & number of afferent (dendritic) synapses
    cell_table.insert(column='total_dendrite_length', value=np.nan, loc=cell_table.shape[1])
    cell_table.insert(column='local_E_syn_count', value=-1, loc=cell_table.shape[1])
    cell_table.insert(column='local_I_syn_count', value=-1, loc=cell_table.shape[1])
    
    for p in proj_names:
        cell_table.insert(column=p + '_count', value=-1, loc=cell_table.shape[1])
    
    for idx, gid in enumerate(gids):
        
        # Query morphology
        nrn = circuit.morph.get(gid)
        cell_table.at[gid, 'total_dendrite_length'] = nm.get('total_length', nrn, neurite_type=nm.BASAL_DENDRITE) + nm.get('total_length', nrn, neurite_type=nm.APICAL_DENDRITE)
        
        # Query local connectivity within given circuit target in isolation
        syn = circuit.connectome.afferent_synapses(gid, properties=[Synapse.PRE_GID, Synapse.POST_BRANCH_TYPE, Synapse.TYPE])
        syn = syn[np.isin(syn[Synapse.PRE_GID], pre_cell_ids)]
        cell_table.at[gid, 'local_E_syn_count'] = np.sum(np.logical_and(syn[Synapse.TYPE] >= 100, np.logical_or(syn[Synapse.POST_BRANCH_TYPE] == NeuriteType.apical_dendrite, syn[Synapse.POST_BRANCH_TYPE] == NeuriteType.basal_dendrite)))
        cell_table.at[gid, 'local_I_syn_count'] = np.sum(np.logical_and(syn[Synapse.TYPE] < 100, np.logical_or(syn[Synapse.POST_BRANCH_TYPE] == NeuriteType.apical_dendrite, syn[Synapse.POST_BRANCH_TYPE] == NeuriteType.basal_dendrite)))
        
        # Query projections
        # | Properties not available: Synapse.POST_BRANCH_TYPE, Synapse.TYPE
        # | => ASSUME: Only one type of synapses, targeting dendrites
        for p in proj_names:
            cell_table.at[gid, p + '_count'] = len(circuit.projection(p).afferent_synapses(gid))
        
        # Progress
        if idx == 0 or np.mod(idx + 1, np.floor(len(gids) / 5).astype(int)) == 0:
            print(f'PROGRESS [GIDs {gids[0]}..{gids[-1]}]: {np.round(100 * (idx + 1) / len(gids)).astype(int)}%', flush=True)
    
    # Compute dendritic synapse density
    cell_table.insert(column='local_E_syn_density', value=cell_table['local_E_syn_count'] / cell_table['total_dendrite_length'], loc=cell_table.shape[1])
    cell_table.insert(column='local_I_syn_density', value=cell_table['local_I_syn_count'] / cell_table['total_dendrite_length'], loc=cell_table.shape[1])
    for p in proj_names:
        cell_table.insert(column=p + '_density', value=cell_table[p + '_count'] / cell_table['total_dendrite_length'], loc=cell_table.shape[1])
    
    return cell_table


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        circuit_config = sys.argv[1]
        
        if len(sys.argv) >= 3:
            circuit_target = sys.argv[2]
        else:
            circuit_target = 'All'
        
        if len(sys.argv) >= 4:
            N_jobs = int(sys.argv[3])
        else:
            N_jobs = 1 # No parallelization
        
        if len(sys.argv) >= 5:
            N_split = int(sys.argv[4])
        else:
            N_split = N_jobs
                
        assert N_split >= N_jobs, 'ERROR: Number of data splits too low for given number of parallel jobs!'
        
        # Process
        t_start = time.time()
        cell_ids_split = create_data_splits(circuit_config, circuit_target, N_split)
        cell_table = pd.concat(Parallel(n_jobs=N_jobs, verbose=100)(delayed(create_cell_table)(circuit_config, circuit_target, cell_ids_split[idx]) for idx in range(N_split)))
        print(f'INFO: Total time elapsed ({cell_table.shape[0]} cells): {time.time() - t_start:.0f}s')
        
        # Save table
        circuit_name = '_'.join(circuit_config.split('/')[-4:]) # Extract circuit name from path
        save_date = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        save_path = f'./cell_table__{circuit_name}__Target_{circuit_target}__{save_date}.pkl'
        
        cell_table.to_pickle(save_path)
        print(f'INFO: Cell table saved to "{save_path}"', flush=True)
        
    else:
        print('Extracts dendritic synapse density (#synapses/um) for each neuron within given circuit target')
        print('Usage: Provide path to circuit config file')
        print('       dendritic_synapse_density PATH_TO_CONFIG <CIRCUIT_TARGET> <#PARALLEL_JOBS> <#DATA_SPLITS>')
