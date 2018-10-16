import tensorflow as tf
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow
import numpy as np


varkeys = ['fire2/expand1x1/kernels', 'fire5/squeeze1x1/kernels', 'fire8/squeeze1x1/kernels', 'fire3/expand3x3/kernels', 'fire7/expand1x1/kernels',
           'fire5/expand3x3/kernels', 'fire4/expand1x1/kernels', 'fire6/expand1x1/kernels', 'fire6/expand3x3/kernels', 'fire3/expand1x1/kernels',
           'fire10/squeeze1x1/kernels', 'fire10/expand1x1/kernels', 'fire9/expand1x1/kernels', 'conv1/kernels', 'fire11/expand3x3/kernels', 'fire6/squeeze1x1/kernels',
           'fire7/squeeze1x1/kernels', 'fire9/expand3x3/kernels', 'conv12/kernels', 'fire8/expand1x1/kernels', 'fire11/expand1x1/kernels', 'fire11/squeeze1x1/kernels',
           'fire2/expand3x3/kernels', 'fire2/squeeze1x1/kernels', 'fire4/squeeze1x1/kernels', 'fire7/expand3x3/kernels', 'fire5/expand1x1/kernels',
           'fire10/expand3x3/kernels', 'fire3/squeeze1x1/kernels', 'fire8/expand3x3/kernels', 'fire9/squeeze1x1/kernels', 'fire4/expand3x3/kernels',
           'fire2/expand1x1/biases', 'fire5/squeeze1x1/biases', 'fire8/squeeze1x1/biases', 'fire3/expand3x3/biases',
           'fire7/expand1x1/biases',
           'fire5/expand3x3/biases', 'fire4/expand1x1/biases', 'fire6/expand1x1/biases', 'fire6/expand3x3/biases',
           'fire3/expand1x1/biases',
           'fire10/squeeze1x1/biases', 'fire10/expand1x1/biases', 'fire9/expand1x1/biases', 'conv1/biases',
           'fire11/expand3x3/biases', 'fire6/squeeze1x1/biases',
           'fire7/squeeze1x1/biases', 'fire9/expand3x3/biases', 'conv12/kernels', 'fire8/expand1x1/biases',
           'fire11/expand1x1/biases', 'fire11/squeeze1x1/biases',
           'fire2/expand3x3/biases', 'fire2/squeeze1x1/biases', 'fire4/squeeze1x1/biases', 'fire7/expand3x3/biases',
           'fire5/expand1x1/biases',
           'fire10/expand3x3/biases', 'fire3/squeeze1x1/biases', 'fire8/expand3x3/biases',
           'fire9/squeeze1x1/biases', 'fire4/expand3x3/biases'
           ]

def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors,
                                     all_tensor_names=False):
  """Prints tensors in a checkpoint file.
  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.
  If `tensor_name` is provided, prints the content of the tensor.
  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
    all_tensors: Boolean indicating whether to print all tensors.
    all_tensor_names: Boolean indicating whether to print all tensor names.
  """

  npyvar = {}
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors or all_tensor_names:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        if key in varkeys:
          print("tensor_name: ", key)
          if 'conv' in key:
              keysplit = key.split('/')
              if keysplit[0] not in npyvar.keys():
                  npyvar[keysplit[0]] = {}
              if keysplit[1] == 'kernels':
                  npyvar[keysplit[0]]['weights'] = np.array(reader.get_tensor(key)).transpose(3,2,0,1)
              if keysplit[1] == 'biases':
                  npyvar[keysplit[0]]['biases'] = np.array(reader.get_tensor(key))
          else:
              keysplit = key.split('/')
              if (keysplit[0]+'/'+keysplit[1]) not in npyvar.keys():
                  npyvar[keysplit[0]+'/'+keysplit[1]] = {}
              if keysplit[2] == 'kernels':
                  npyvar[keysplit[0]+'/'+keysplit[1]]['weights'] = np.array(reader.get_tensor(key)).transpose(3,2,0,1)
              if keysplit[2] == 'biases':
                  npyvar[keysplit[0]+'/'+keysplit[1]]['biases'] = np.array(reader.get_tensor(key))
          #else:
          #    if all_tensors:
          #        print(np.array(reader.get_tensor(key)).transpose(3,2,0,1)[0,:,:,:])
    elif not tensor_name:
      print(reader.debug_string().decode("utf-8"))
    else:
      print("tensor_name: ", tensor_name)
      print(reader.get_tensor(tensor_name))
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")
    if ("Data loss" in str(e) and
        (any([e in file_name for e in [".index", ".meta", ".data"]]))):
      proposed_file = ".".join(file_name.split(".")[0:-1])
      v2_file_error_template = """
      It's likely that this is a V2 checkpoint and you need to provide the filename
      *prefix*.  Try removing the '.' and extension.  Try:
      inspect checkpoint --file_name = {}"""
      print(v2_file_error_template.format(proposed_file))
  np.save('/home/siiva/RUDY/SIIVA/GoalCam/annotations/shanghaioffice/ALL/models/train/squeezedet_half.npy', npyvar)
latest_ckp = tf.train.latest_checkpoint('/home/siiva/RUDY/SIIVA/GoalCam/annotations/shanghaioffice/ALL/models/train/') #('/home/rudy/TITIP/ONDEVICESAI/squeezeDet/data/squeezeDet/train/')
print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')



# # npy = np.load('/home/rudy/Downloads/squeezeDet.npy', encoding='latin1')
# npy_model = np.load('/home/rudy/Downloads/squeezeDet.npy', encoding='latin1').item()
# print(np.array(npy_model['conv12']['weights']).shape)
# #print(np.array(npy_model['conv12']['biases']))

# print(npy_model.keys())

npy_model1 = np.load('/home/siiva/RUDY/SIIVA/GoalCam/annotations/ALL_zoom/models/train/squeezedet_half.npy', encoding='latin1').item()
print(npy_model1.keys())
print(np.array(npy_model1['conv12']['weights']).shape)
#print(np.array(npy_model1['conv12']['biases']))

#
# for key in npy_model.keys():
#     print(key, np.sum(
#         np.array(npy_model[key]['weights'])[0, :, :, :] - np.array(npy_model1[key]['weights'])[0, :, :, :]))
#
#     assert np.sum(np.array(npy_model['conv1']['weights'])[0,:,:,:] - np.array(npy_model1['conv1']['weights'])[0,:,:,:]) == 0, "not same"
