from utils.coffea_processors.lz4_processor import LZ4Processing 

class LZ4_PairedTaggerProcessor(LZ4Processing):
    
    """
    class LZ4Processing(DataPreprocessing_BaseClass):
        def callColumnAccumulator(self, output, events, flag, **kwargs):
        ....
    return (
            global_arr[nan_mask],
            cpf_arr[nan_mask],
            npf_arr[nan_mask],
            vtx_arr[nan_mask],
            truth_arr[nan_mask],
            process[nan_mask],
        )
    """
    
    def callColumnAccumulator(self, output, events, flag, **kwargs):
        global_arr, cpf_arr, npf_arr, vtx_arr, lt_arr, truth_arr, process = super().callColumnAccumulator(output, events, flag, **kwargs)
        
        #print('----- Inside PairedTaggerProcessor -----')
        #print('global_arr.shape:', global_arr.shape)
        
        selection_mask = (
                    (~global_arr['includesIsoLepJet'].astype(bool)) &
                    (global_arr['MC_gen_flav'] == 0) &
                    (truth_arr['label_ll'] == 1)
                ) | (
                    truth_arr['label_cc'] == 1
                ) | (
                    truth_arr['label_bb'] == 1
                )
        #print('selection_mask.shape:', selection_mask.shape)
        #print('selection_mask.sum():', selection_mask.sum())
        #print('global_arr.shape:', global_arr[selection_mask].shape)
        # raise Exception('Stop here')
        return (
            global_arr[selection_mask],
            cpf_arr[selection_mask],
            npf_arr[selection_mask],
            vtx_arr[selection_mask],
            lt_arr[selection_mask],
            truth_arr[selection_mask],
            process[selection_mask],
        )