# Retrieve the package location
import os
import snn
import inspect
package_path = os.path.dirname(inspect.getfile(snn))

import tensorflow.compat.v1 as tf  
tf.disable_v2_behavior()  
config = tf.ConfigProto(intra_op_parallelism_threads=int(os.environ.get("TF_NUM_THREADS", "0")), inter_op_parallelism_threads=int(os.environ.get("TF_NUM_THREADS", "0"))) 

