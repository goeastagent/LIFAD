from mmrnn import *
from DataManager import *


import numpy as np

dm = DataManager()

cog, y_cog, RIDs_cog, seqlen_cog = dm.generate_cog(9)
csf, y_csf, RIDs_csf, seqlen_csf = dm.generate_csf(3)
MRI, y_MRI, RIDs_MRI, seqlen_MRI = dm.generate_MRI()
demo, y_demo, RIDs_demo, seqlen_demo = dm.generate_demo()

overlap_RIDs = dm.generate_overlapRIDs([RIDs_cog,RIDs_csf,RIDs_MRI,RIDs_demo])
trainRIDs, testRIDs = dm.generate_training_test_RIDs(overlap_RIDs)

m = MMRNN()
m.append_component('cog',cog.shape[2], 3,cog.shape[1])
m.append_component('csf',csf.shape[2], 6,csf.shape[1])
m.append_component('MRI',MRI.shape[2], 4,MRI.shape[1])
m.append_component('demo',demo.shape[2], 5,demo.shape[1])

m.append_data('cog',RIDs_cog, cog, y_cog, seqlen_cog)
m.append_data('csf',RIDs_csf, csf, y_csf, seqlen_csf)
m.append_data('MRI',RIDs_MRI, MRI, y_MRI, seqlen_MRI)
m.append_data('demo',RIDs_demo, demo, y_demo, seqlen_demo)

m.append_test_overlapIDs(testRIDs)
m.append_training_overlapIDs(trainRIDs)
m.build_integrative_network()
m.training(32)

print(m.get_coefficient)

#print(m.evaluate_balanced_accuracy())

# m.single_feature_extraction()
# m.integrative_feature_extraction()

