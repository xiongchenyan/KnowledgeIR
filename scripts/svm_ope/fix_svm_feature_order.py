from knowledge4ir.utils import load_svm_feature, dump_svm_feature
import sys

if 3 != len(sys.argv):
    print "svm in + out"
    sys.exit()

l = load_svm_feature(sys.argv[1])
dump_svm_feature(l, sys.argv[2])