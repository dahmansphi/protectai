
# Copyright (c) 2024 Dr. Deniz Dahman's <denizdahman@gmail.com>
 
# This file may be used under the terms of the GNU General Public License
# version 3.0 as published by the Free Software Foundation and appearing in
# the file LICENSE included in the packaging of this file.  Please review the
# following information to ensure the GNU General Public License version 3.0
# requirements will be met: http://www.gnu.org/copyleft/gpl.html.
# 
# If you do not wish to use this file under the terms of the GPL version 3.0
# then you may purchase a commercial license.  For more information contact
# denizdahman@gmail.com.
# 
# This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
# WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

# Support the author of this package on:.#
# https://patreon.com/user?u=118924481
# https://www.youtube.com/@dahmansphi 
# https://dahmansphi.com/subscriptions/


import os
import time
import math
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class ProtectAI:

  __dictNumberOfImgsPerCls = {}
  __control_path_status = None
  __control_imgs_status = None

  def __init__(self) -> None:
     pass

  def _control_path_spec(self, paths_folder):
    '''This function control the spec of the paths both training and testing
      make sure they have the same folders where each of which represents the class
    '''
    test_dict_holder = None

    for _index in range(len(paths_folder)):
      path_folder = paths_folder[_index]
      print(f"the tool is controlling **{os.path.basename(path_folder)}** folder")

      _entries = os.listdir(path_folder)
      _folders = [folder for folder in _entries if os.path.isdir(os.path.join(path_folder,folder))]

      if _index == 0:
        _dic_folders_details = {}

        _folder_name = os.path.basename(path_folder)
        _num_cls_in_folders = len(_folders)
        _dic_folders_details[_folder_name] = [_num_cls_in_folders, _folders]
        test_dict_holder = _dic_folders_details

      else:
        _dic_folders_details = {}
        _key_test_dic = list(test_dict_holder.keys())[0]

        _folder_name = os.path.basename(path_folder)
        _num_cls_in_folders = len(_folders)
        # isEqual = test_dict_holder[_key_test_dic] == [_num_cls_in_folders, _folders]
        isEqual_cond = None
        for cls in [_num_cls_in_folders, _folders][1]:
          isEqual_cond = cls in test_dict_holder[_key_test_dic][1]

        if isEqual_cond:
          self.__control_path_status = True
          continue
        else:
          self.__control_path_status = False
          break


  def _explor_path(self, paths_folder):
    '''This function controls the status and valid of the imgs in both training
      and testing folders; it counts on valid jpeg format and L mode type img
    '''

    self._control_path_spec(paths_folder)
    if self.__control_path_status:

      print(f"Folders controlling is done; Now the tool is checking the contents status")

      for path_folder in paths_folder:

        _entries = os.listdir(path_folder)
        _folders = [folder for folder in _entries if os.path.isdir(os.path.join(path_folder,folder))]

        _folder_name = os.path.basename(path_folder)
        _num_cls_in_folders = len(_folders)

        self.__dictNumberOfImgsPerCls[_folder_name] = []

        print("")
        print(f"The tool is checking the contents status of {_folder_name} Folder")

        for folder in _folders:

          print(f"The tools is checking the {folder} inside the {_folder_name}")

          _class_folder_path = os.path.join(path_folder, folder)
          _imgs = os.listdir(_class_folder_path)

          _imgs = [im for im in _imgs if im.lower().endswith(('.jpg', '.jpeg'))]

          print(f"The tool is controlling {len(_imgs)} inside the {folder}")
          _count = 0
          print("")
          for img in _imgs:

            if _count < 150:
              print("*", end='')
            else:
              print("*", end='\r')
              _count = 0

            _img_full_path = os.path.join(_class_folder_path, img)
            try:
              fobj = open(_img_full_path, "rb")
              is_jfif = b"JFIF" in fobj.peek(10)
            finally:
              fobj.close()
            if not is_jfif:
              os.remove(_img_full_path)
            else:
              _img = Image.open(_img_full_path)
              if _img.mode != 'L':
                _new_name = 'N0' + img
                _new_img = _img.convert('L')
                _new_img.save(os.path.join(_class_folder_path, _new_name))
                os.remove(_img_full_path)

            _count += 1

          self.__dictNumberOfImgsPerCls[_folder_name].append((folder, len(_imgs)))
          print("")

      self.__control_imgs_status = True
      print("The tool is done controlling the contents and folders")

    else:
      print("Number of classess are not consistent in both folders[training, testing] which you have provided")

  def set_path_models(self, paths_folder):
    '''This function set the preparation for the paths [training, testing] provided'''

    msg = "*" * 50
    print(msg, "Welcome to ClassifyModel_ tool to create DeepLearning Predictive Model by: Dr. Deniz Dahman", sep='\n')
    print("Visit dahmansphi.com for information", msg, sep='\n')

    self._explor_path(paths_folder=paths_folder)
    if self.__control_imgs_status:
      print("here is the report:")
      for k,v in self.__dictNumberOfImgsPerCls.items():
        print(f"The {k} has {v}")
    else:
      print("Your imgs are not valid for either training or testing")


  def make_model(self):
    '''This function create the model skeleton, using standard 180 Resl.'''

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(180, 180, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    # opt = SGD(learning_rate=0.001, momentum=0.9)
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

  def train_test_model(self, model, paths_folder):
    '''This tool use the train and test data set to train then test model if ok'''

    trainPath = paths_folder[0]
    testPath = paths_folder[1]

    # create data generators
    train_datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # prepare iterators
    train_it = train_datagen.flow_from_directory(trainPath, color_mode="grayscale", class_mode='binary', batch_size=64, target_size=(180, 180))
    test_it = test_datagen.flow_from_directory(testPath, color_mode="grayscale", class_mode='binary', batch_size=64, target_size=(180, 180))

    # fit and test model
    history = model.fit(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=1)

    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))

  def produce_model(self, model,paths_folder):
    '''Once the result from train_test ok then one can save and create the model'''

    trainPath = paths_folder[0]
    saveModelPath = paths_folder[1]

    # create data generators
    train_datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    # prepare iterators
    train_it = train_datagen.flow_from_directory(trainPath, color_mode="grayscale", class_mode='binary', batch_size=64, target_size=(180, 180))
    # fit model
    history = model.fit(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=0)
    # save the model

    time_stamp = str(int(time.time()))
    saved_model_name = 'CLASSIFY_M_01' + time_stamp + '.keras'
    model_to_save = os.path.join(saveModelPath, saved_model_name)
    model.save(model_to_save)

    #  report the model acc
    acc = history.history['accuracy']
    print(f'accuracy is: {acc}')

  def predict(self, model_path, img_path, inner_case = False):
    '''This function will perform the task of prediction'''

    _predictions_inner_case = []

    model = load_model(model_path)
    for img in os.listdir(img_path):
      full_img_path = os.path.join(img_path, img)
      _cls_name = os.path.basename(img_path)
      _img_name = img

      try:
        fobj = open(full_img_path, "rb")
        is_jfif = b"JFIF" in fobj.peek(10)
      finally:
        fobj.close()
      if not is_jfif:
        # os.remove(full_img_path)
        pass
      else:
        _img = Image.open(full_img_path)
        if _img.mode != 'L':
          _new_name = 'N0' + img
          _new_img = _img.convert('L')
          _new_img.save(os.path.join(img_path, _new_name))
          os.remove(full_img_path)

        filename = full_img_path

        img = load_img(filename, target_size=(180, 180, 1), color_mode='grayscale')
        # # convert to array
        img = img_to_array(img)
        # # img.shape
        # # # reshape into a single sample with 1 channels
        img = img.reshape(1, 180, 180, 1)
        # (model.predict(img)[0][0])
        predictions = model.predict(img, verbose = 0)[0][0]

        if inner_case == False:
          print(f"This image is {100 * (1 - predictions):.2f}% Normal and {100 * predictions:.2f}% Pnemonia.")
        else:
          _predictions_inner_case.append([predictions, _img_name, _cls_name])

    if inner_case == True:
      return _predictions_inner_case

  def update_trained_model_with_test(self, model, paths):
    '''This function perform train update on a  trained model with test no production'''
    model = load_model(model)
    self.train_test_model(model=model, paths_folder=paths)

  def update_trained_model_produce(self, model, paths_produce):
    '''This function perform train update on a  trained model without test finall for producing a model'''
    model = load_model(model)
    self.produce_model(model=model, paths_folder=paths_produce)

  # *****************************************
  def protect_me(self, paths, save_path_norms):
    '''This function creates the norm of the right confident set'''
    if self.__control_imgs_status:
      print("This tool is designed by Dr. Deniz Dahman for simulation and education purpose ONLY")
      star_ms = "*" * 20
      print(star_ms)
      _num_cls = None
      _name_cls = None
      _isNumCls = None
      _isNameCls = None

      _pics_db = None
      for _index in range(len(paths)):
        path = paths[_index]
        if _index == 0:
          _folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
          _num_cls = len(_folders)
          _name_cls = _folders
        else:
          _folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
          _isNumCls = len(_folders) == _num_cls
          for _fold in _folders:
            if _fold in _name_cls:
              _isNameCls = True
            else:
              _isNameCls =False
              break

      if _isNumCls and _isNameCls:
        _pics_db = [[] for _ in range(_num_cls)]

        for path in paths:
          for _inx in range(len(_name_cls)):
            _list = _pics_db[_inx]
            cls = _name_cls[_inx]

            _folder_full_path = os.path.join(path, cls)

            _imgs = os.listdir(_folder_full_path)
            # _imgs = [img for img in os.listdir(_folder_full_path)]
            _imgs = [im for im in _imgs if im.lower().endswith(('.jpg', '.jpeg'))]

            for _img in _imgs:
              _img_full_path = os.path.join(_folder_full_path, _img)
              _list.append(_img_full_path)

        _pic_flatten_list = [[] for _ in range(_num_cls)]
        _pic_norms_list = [[] for _ in range(_num_cls)]

        for _inx in range(len(_pics_db)):
          _counter = 0
          _db_cls = _pics_db[_inx]
          _db_flatten = _pic_flatten_list[_inx]
          _db_norms = _pic_norms_list[_inx]
          _cls_name = _name_cls[_inx]

          print("")
          print(f"Now we are norming the imgs for the {_cls_name} class")

          for _innn in range(len(_db_cls)):
            _img = _db_cls[_innn]

            _img_format = Image.open(_img)
            _new_img_transformed = _img_format.resize((180,180))
            _new_img_transformed_arr = np.array(_new_img_transformed)

            _norm = np.linalg.norm(_new_img_transformed_arr)
            _new_img_transformed_arr = _new_img_transformed_arr/_norm

            _new_img_flatten = _new_img_transformed_arr.flatten()
            _db_flatten.append(_new_img_flatten)
            _db_norms.append(_norm)

            # _img_arr = np.array(_img_format)
    #         _norm = np.linalg.norm(_img_arr)
    #         _db_flatten.append(_norm)

            if _counter < 100:
              print("*", end='')
            else:
              print("*", end='\r')
              _counter = 0

            _counter += 1
          print("")

        for _indx in range(len(_pic_flatten_list)):

          _mat_flatten = _pic_flatten_list[_indx]
          _mat_norms = _pic_norms_list[_indx]
          _cls_name = _name_cls[_indx]

          _mat_flatten_arr = np.array(_mat_flatten)
          _mat_norms_arr = np.array(_mat_norms)

          _mat_flatten_first_row_out = _mat_flatten_arr[1:,:]
          _flatten_first_row_out = _mat_flatten_arr[:1,:]

          _capture_norm_alpha = np.sqrt((_flatten_first_row_out - _mat_flatten_first_row_out)**2)
          _norm_alpha = np.linalg.norm(_capture_norm_alpha)
          _norm_alpha = np.array(_norm_alpha)

          time_stamp = str(int(time.time()))
          saved_flatten_name = 'flatten_' + time_stamp + "_" +_cls_name +'.npy'
          saved_norms_name = 'norms_' + time_stamp + "_" +_cls_name +'.npy'
          saved_alpha_name = 'alpha_' + time_stamp + "_" + _cls_name + '.npy'

          _path_to_save_flatten = os.path.join(save_path_norms, saved_flatten_name)
          _path_to_save_norms = os.path.join(save_path_norms, saved_norms_name)
          _path_to_save_alpha = os.path.join(save_path_norms, saved_alpha_name)

          np.save(file=_path_to_save_flatten, arr=_mat_flatten_arr)
          np.save(file=_path_to_save_norms, arr=_mat_norms_arr)
          np.save(file=_path_to_save_alpha, arr=_norm_alpha)


          print(f"the culture of the {_cls_name} of alpha {_norm_alpha} is created and saved as {saved_flatten_name} and {saved_norms_name} on the provided save path")

    else:
      print("You must first call on set_path_models() function then you can protect your model")

  def inspect_attack_one(self, paths_to_inpect, paths_to_norms):

    self.__control_imgs_status = True

    if self.__control_imgs_status:

      # _norms_names = [_norm for _norm in os.listdir(paths_to_norm) if _norm.lower().endswith('.npy')]
      _norms_names = [_norm for _norm in os.listdir(paths_to_norms) if _norm.lower().startswith('norms')]
      _flatten_names = [_norm for _norm in os.listdir(paths_to_norms) if _norm.lower().startswith('flatten')]
      _alphas_names = [_norm for _norm in os.listdir(paths_to_norms) if _norm.lower().startswith('alpha')]

      _cls_names = [folder for folder in os.listdir(paths_to_inpect) if os.path.isdir(os.path.join(paths_to_inpect, folder))]
      _norm_has_cls = None

      for _cls in _cls_names:

        _internal_check = None
        for _norm_name in _norms_names:
          isNormIn = _cls in _norm_name
          if isNormIn:
            _internal_check = True
            break

        if _internal_check != True:
          print("The norms has no match with the inspected folder")
          break
        else:
          _norm_has_cls = True

      if _norm_has_cls:
        _norms_dic = {}
        _flatten_dic = {}
        _alpha_dic = {}

        for _inx in range(len(_norms_names)):

          _norm_name = _norms_names[_inx]
          _flatten_name = _flatten_names[_inx]
          _alpha_name = _alphas_names[_inx]

          _norm_id = None
          _flatten_id = None
          _alpha_id = None

          for _cls in _cls_names:
            if _cls in _norm_name:
              _norm_id = _cls
              _flatten_id = _cls
              _alpha_id = _cls

          if _norm_id and _flatten_id and _alpha_id:

            _norm_name_full_path = os.path.join(paths_to_norms, _norm_name)
            _flatten_name_full_path = os.path.join(paths_to_norms, _flatten_name)
            _alpha_name_full_path = os.path.join(paths_to_norms, _alpha_name)

            _arr_np_norm = np.load(_norm_name_full_path)
            _arr_np_flatten = np.load(_flatten_name_full_path)
            _arr_np_alpha = np.load(_alpha_name_full_path)

            _norms_dic[_norm_id] = _arr_np_norm
            _flatten_dic[_flatten_id] = _arr_np_flatten
            _alpha_dic[_alpha_id] = _arr_np_alpha
        # ***************************************************

        for _cls_name in _cls_names:

          isClsNorm = _cls_name in _norms_dic.keys()

          if isClsNorm:

            _norm_vals = _norms_dic[_cls_name]
            _flatten_vals = _flatten_dic[_cls_name]
            _alpha_vals = _alpha_dic[_cls_name]

            print(f"This is class {_cls_name} of alpha {_alpha_vals}")
            print("***************************")

          #   _norm_avg_mat = np.linalg.norm(_norm_vals)
          #   # norm the mat of transofrmed
          #   _norm_vals = _norm_vals/_norm_avg_mat
          # #   _norm_avg_mean = np.mean(_norm_vals)

            _cls_full_path = os.path.join(paths_to_inpect, _cls_name)
            _imgs = os.listdir(_cls_full_path)
            _imgs = [im for im in _imgs if im.lower().endswith(('.jpg', '.jpeg'))]

            for _img in _imgs:
              _dis_arr = []

              _img_full_path = os.path.join(_cls_full_path, _img)
              _img_format = Image.open(_img_full_path)
              _new_inspect_img = _img_format.resize((180,180))
              _new_inspect_img_arr = np.array(_new_inspect_img)

              _new_inspect_img_norm = np.linalg.norm(_new_inspect_img_arr)
              _new_inspect_img_arr = _new_inspect_img_arr/_new_inspect_img_norm

              _new_inspect_img_flatten = _new_inspect_img_arr.flatten()
          #     _new_inspect_img_flatten = _new_inspect_img_flatten/_norm_avg_mat
              _dist = np.sqrt((_flatten_vals - _new_inspect_img_flatten)**2) # num_imgsX32400 (180x180)
              _norm_dist = np.linalg.norm(_dist) # norm num_imgsX32400

              _diff_from_alpha = np.sqrt((_norm_dist - _alpha_vals)**2)

              # _dis_arr.append(_norm_dist)
          #     _dis_arr = np.array(_dis_arr)
          #     _min_dis = np.min(_dis_arr)
              # _norm_org_avg = np.mean(_norm_vals)
              # _diff_avg_norms = np.sqrt((_norm_org_avg - _new_inspect_img_norm)**2)
              _warn = None
              if _diff_from_alpha > 3:
                _warn = "***WARNING***"
              else:
                _warn = ""

              print('img_norm',_norm_dist,'Name' ,_img, 'distance from alph', _diff_from_alpha, _warn)

          print("***************************")
  def inspect_attack_two(self, paths_to_inpect, model_verified):

    _folders = [folder for folder in os.listdir(paths_to_inpect) if os.path.isdir(os.path.join(paths_to_inpect, folder))]
    _folders.sort()

    _prediction_list = {}

    for _folder in _folders:
      _cls_name = _folder
      _prediction_list[_cls_name] = []
      _full_cls_path = os.path.join(paths_to_inpect, _cls_name)
      _resl = self.predict(img_path=_full_cls_path,model_path=model_verified, inner_case=True)
      _prediction_list[_cls_name].append(_resl)

    for _inx in range(len(_folders)):
      _cls_name = list(_prediction_list.keys())[_inx]
      _other_cls_name = [_name for _name in _folders if _name != _cls_name]
      list_verify = _prediction_list[_cls_name][0]
      for _reslt in list_verify:
        _pred_res = _reslt[0]
        if _pred_res != _inx:
          _img_name = _reslt[1]
          print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
          print(f"Warning imgae {_img_name} is in the folder of class {_cls_name}")
          print(f"It appears as if it's belong the other class {_other_cls_name[0]}!")
          print(f"if you are sure it's in the right folder {_cls_name} under the path {paths_to_inpect}")
          print("then it is ok, otherwise you might have experienced swap poisoning attack")
          print("__________________________________________________________")
