# For whatever reason, this no longer works (but it did on SVD++ before...)
# Note that you need to compile using Makefile_blend (rename it to Makefile) and
# then do a make to get the below to work...
import cppyy

cppyy.include("predict.hpp")
cppyy.load_library("readfile")
cppyy.load_library("svdplusplus")

from cppyy.gbl import Predict
from cppyy.gbl import SVDPlusPlus
from cppyy.gbl import Data

p = Predict()
p.run_model()
