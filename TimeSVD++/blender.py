import cppyy

cppyy.include("predict.hpp")
cppyy.load_library("readfile")
cppyy.load_library("svdplusplus")

from cppyy.gbl import Predict
from cppyy.gbl import SVDPlusPlus
from cppyy.gbl import Data

p = Predict()
p.run_model()
