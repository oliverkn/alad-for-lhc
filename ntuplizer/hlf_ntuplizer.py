from ntuplizer.ntuplizer import *


class HT_module(AbstractQuantityModule):
    """
       take all double_ak5PFJets_sigma_RECO elements with pT>30 and
       compute the scalar sum of their pTs
    """

    def compute(self, values):
        pt = values['recoPFJets_ak5PFJets__RECO.obj.pt_']
        HT = np.zeros(pt.shape[0])
        for i in range(pt.shape[0]):
            pt_f = np.where(pt[i] > 30, pt[i], 0)
            HT[i] = np.sum(pt_f)

        return HT.reshape((-1, 1))

    def get_keys(self):
        return ['recoPFJets_ak5PFJets__RECO.obj.pt_']

    def get_size(self): return 1

    def get_names(self): return 'HT'


ntuplizer = Ntuplizer()
ntuplizer.register_quantity_module(HT_module())

ntuplizer.convert('/home/oliverkn/pro/real_data_test/test.root', '/home/oliverkn/pro/real_data_test/test.npy')
