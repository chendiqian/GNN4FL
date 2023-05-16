import numpy as np
import sys
from sys import getsizeof
import json
import fractions
import math
import random
import inspect
import datetime
import time
import pandas as pd

import phe as paillier
from phe import EncodedNumber, EncryptedNumber
from phe.util import invert, powmod, mulmod, getprimeover, isqrt
from collections import defaultdict
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mnist_cnn import MNIST_CNN

LOG2_BASE = 4.0
FLOAT_MANTISSA_BITS = sys.float_info.mant_dig
BASE = 16

class myThread (threading.Thread):
    def __init__(self, threadID, matrix, encryption):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.encryption = encryption
        self.matrix = matrix

    def run(self):
        # Get lock to synchronize threads
        threadLock.acquire()
        # print_time(self.name, self.counter, 3)
        model_projection_paillier(self.threadID, self.encryption, self.matrix)
        # Free lock to release next thread
        threadLock.release()

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def l_function(x, p):
    """Computes the L function as defined in Paillier's paper. That is: L(x,p) = (x-1)/p"""
    return (x - 1) // p

def h_function(public_g, x, xsquare):
    """Computes the h-function as defined in Paillier's paper page 12,
    'Decryption using Chinese-remaindering'.
    """
    return invert(l_function(pow(public_g, x - 1, xsquare),x), x)

def model_encryption(model_dict_weights, public_n, r_value=None, printF=None):
    """
    :param model_dict_weights:
        {key: X, ...}
        type(x) == torch.Tensor
    :param public_n:
    :return:
        model_dict_encryption
            {key: {ciphertext: X, exponent: X}, ...}
            type(x) == np.ndarray
    """
    print("############################  MODEL ENCRYPTION ##########################################")

    model_dict_encryption = defaultdict(dict)
    if printF:
        print("------------------ Here goes the process for public key -------------------------")
    public_nsquare = public_n * public_n

    for key, weight_torch in model_dict_weights.items():
        weight = weight_torch.numpy().copy()
        weight_shape = weight.shape
        weight_size = weight.size

        if printF:
            print("------------------ Here goes the process for encrypt", key, "-------------------------")
        # ------------------------- Encoded -------------------------  #
        # class EncodedNumber(object):
        #     def __init__(self, public_key, encoding, exponent):
        #         self.public_key = public_key
        #         self.encoding = encoding
        #         self.exponent = exponent
        _, bin_flt_exponent = np.frexp(weight)
        bin_lsb_exponent = bin_flt_exponent - FLOAT_MANTISSA_BITS
        exponent = np.floor(bin_lsb_exponent / LOG2_BASE)
        int_rep_float = np.around(weight * BASE ** -exponent)
        # bias_int_rep_float_ = np.around(bias * BASE ** -bias_exponent)
        int_rep = int_rep_float.astype(int)
        encoding = int_rep % public_n
        model_dict_encryption[key]["exponent"] = exponent

        # ------------------------- Encrypt_encoded -------------------------  #
        # class EncryptedNumber(object):
        #     def __init__(self, public_key, ciphertext, exponent=0):
        #         self.public_key = public_key
        #         self._EncryptedNumber__ciphertext = ciphertext
        #         self._EncryptedNumber__is_obfuscated = False
        #         self.exponent = exponent

        # if public_n - public_max_int <= plaintext < self.n:
        #     # Very large plaintext, take a sneaky shortcut using inverses
        #     neg_plaintext = self.n - plaintext  # = abs(plaintext - nsquare)
        #     # avoid using gmpy2's mulmod when a * b < c
        #     neg_ciphertext = (self.n * neg_plaintext + 1) % self.nsquare
        #     nude_ciphertext = invert(neg_ciphertext, self.nsquare)
        # else:
        #     # we chose g = n + 1, so that we can exploit the fact that
        #     # (n+1)^plaintext = n*plaintext + 1 mod n^2
        #     nude_ciphertext = (self.n * plaintext + 1) % self.nsquare

        # !!! Rewriting !!! #
        neg_plaintext = public_n - encoding  # = abs(plaintext - nsquare)
        neg_ciphertext = (public_n * neg_plaintext + 1) % public_nsquare
        neg_ciphertext_flatten = neg_ciphertext.flatten()
        nude_ciphertext_flatten = np.array([invert(i, public_nsquare) for i in neg_ciphertext_flatten])
        raw_encrypted_number_flatten = nude_ciphertext_flatten % public_nsquare

        if r_value:
            r_flatten = [random.randrange(1, public_n) for i in range(weight_size)]
            r_pow_flatten = np.array([pow(i, public_n, public_nsquare) for i in r_flatten])
            ciphertext_flatten = raw_encrypted_number_flatten * r_pow_flatten % public_nsquare
            ciphertext = ciphertext_flatten.reshape(weight_shape)
        else:
            ciphertext = raw_encrypted_number_flatten.reshape(weight_shape)

        model_dict_encryption[key]["ciphertext"] = ciphertext

    return model_dict_encryption

def model_encryption_paillier(model_dict_weights, public_n):
    """
    :param model_dict_weights:
        {key: X, ...}
        type(x) == torch.Tensor
    :param public_n:
    :return:
        model_dict_encryption
            {key: {np.ndarray}, ...}
    """
    print("############################  MODEL ENCRYPTION ##########################################")

    model_dict_encryption_paillier = defaultdict(list)
    public_key = paillier.PaillierPublicKey(public_n)

    for key, weight_torch in model_dict_weights.items():
        weight = weight_torch.numpy().copy()
        weight_shape = weight.shape
        weight_size = weight.size

        weight_flatten = weight.flatten().tolist()
        ciphertext_flatten = np.array([public_key.encrypt(i) for i in weight_flatten])
        ciphertext = ciphertext_flatten.reshape(weight_shape)

        model_dict_encryption_paillier[key] = ciphertext

    return model_dict_encryption_paillier

def model_decryption(model_dict_encryption, public_n, private_p, private_q, printF=None):
    """
    :param model_dict_encryption:
        {key: {ciphertext: X, exponent: X}, ...}
        type(x) == np.ndarray
    :param public_n:
    :param private_p:
    :param private_q:
    :return:
        model_dict_decryption
            {key: X, ...}
            type(x) == np.ndarray

    """
    print("############################  MODEL DECRYPTION ##########################################")

    model_dict_decryption = dict()
    if printF:
        print("------------------ Here goes the process for private key -------------------------")
    public_max_int = public_n // 3 - 1
    public_nsquare = public_n * public_n
    public_g = public_n + 1
    private_qsquare = private_q * private_q
    private_psquare = private_p * private_p
    private_p_inverse = invert(private_p, private_q)
    private_hp = h_function(public_g, private_p, private_psquare)
    private_hq = h_function(public_g, private_q, private_qsquare)

    for key, item in model_dict_encryption.items():
        weight = item["ciphertext"].copy()
        exponent = item["exponent"].copy()
        weight_shape = weight.shape
        weight_size = weight.size

        if printF:
            print("------------------ Here goes the process for decrypt", key, "-------------------------")
        # ------------------------- Decrypt_encoded -------------------------  #
        # class EncryptedNumber(object):
        #     def __init__(self, public_key, ciphertext, exponent=0):
        #         self.public_key = public_key
        #         self._EncryptedNumber__ciphertext = ciphertext
        #         self._EncryptedNumber__is_obfuscated = False
        #         self.exponent = exponent
        ciphertext = weight.flatten()
        decrypt_to_p_l_function = np.array(
            [l_function(pow(i, private_p - 1, private_psquare), private_p) for i in ciphertext])
        decrypt_to_p = decrypt_to_p_l_function * private_hp % private_p
        decrypt_to_q_l_function = np.array(
            [l_function(pow(i, private_q - 1, private_qsquare), private_q) for i in ciphertext])
        decrypt_to_q = decrypt_to_q_l_function * private_hq % private_q
        decrypted_encoded_flatten = decrypt_to_p + (
        ((decrypt_to_q - decrypt_to_p) * private_p_inverse % private_q)) * private_p
        decrypted_encoded = decrypted_encoded_flatten.reshape(weight_shape)

        # ------------------------- Decoded -------------------------  #
        # class EncodedNumber(object):
        #     def __init__(self, public_key, encoding, exponent):
        #         self.public_key = public_key
        #         self.encoding = encoding
        #         self.exponent = exponent
        mantissa = np.where(decrypted_encoded <= public_max_int, decrypted_encoded,
                                 decrypted_encoded - public_n)
        decrypted = np.where(exponent >= 0, mantissa * BASE ** exponent,
                                  mantissa / BASE ** -exponent)

        model_dict_decryption[key] = decrypted

    return model_dict_decryption

def model_decryption_paillier(model_dict_encryption, public_n, private_p, private_q):
    """

    :param model_dict_encryption:
            {key: {ciphertext: X, exponent: X}, ...}
            type(x) == np.ndarray
    :param public_n:
    :param private_p:
    :param private_q:
    :return:
    """

    print("############################  MODEL DECRYPTION ##########################################")

    model_dict_paillier = defaultdict(list)
    public_key = paillier.PaillierPublicKey(public_n)
    private_key = paillier.PaillierPrivateKey(public_key, private_p, private_q)

    for key, item in model_dict_encryption.items():
        weight = item["ciphertext"].copy()
        exponent = item["exponent"].copy()
        plaintext_shape = weight.shape
        plaintext_size = weight.size
        weight_flatten = weight.flatten()
        exponent_flatten = exponent.flatten()
        plaintext_flatten = np.array([private_key.decrypt(EncryptedNumber(public_key, weight_flatten[i], exponent_flatten[i])) for i in range(plaintext_size)])
        ciphertext = plaintext_flatten.reshape(plaintext_shape)

        model_dict_paillier[key] = ciphertext

    return model_dict_paillier

def model_projection(model_encryption_flatten, projection_matirx, public_n):
    """
        :param model_dict_encryption:
            {key: {ciphertext: X, exponent: X}, ...}
            type(x) == np.ndarray
        :return:
            projection_dict_encryption
                {key: {ciphertext: X, exponent: X}, ...}
                type(x) == np.ndarray
    """
    matrix_shape = projection_matirx.shape
    matrix_size = projection_matirx.size
    public_nsquare = public_n * public_n

    # ------------------------- Encoded -------------------------  #
    # class EncodedNumber(object):
    #     def __init__(self, public_key, encoding, exponent):
    #         self.public_key = public_key
    #         self.encoding = encoding
    #         self.exponent = exponent
    begin_time = time.time()
    _, bin_flt_exponent = np.frexp(projection_matirx)
    bin_lsb_exponent = bin_flt_exponent - FLOAT_MANTISSA_BITS
    exponent = np.floor(bin_lsb_exponent / LOG2_BASE)
    int_rep_float = np.around(projection_matirx * BASE ** -exponent)
    # bias_int_rep_float_ = np.around(bias * BASE ** -bias_exponent)
    int_rep = int_rep_float.astype(int)
    encoding = int_rep % public_n

    neg_plaintext = public_n - encoding  # = abs(plaintext - nsquare)
    neg_ciphertext = (public_n * neg_plaintext + 1) % public_nsquare
    neg_ciphertext_flatten = neg_ciphertext.flatten()
    nude_ciphertext_flatten = np.array([invert(i, public_nsquare) for i in neg_ciphertext_flatten])
    raw_encrypted_number_flatten = nude_ciphertext_flatten % public_nsquare

    matrix_ciphertext = raw_encrypted_number_flatten.reshape(matrix_shape)
    print(f"time for 前摇: {time.time() - begin_time}")

    begin_time = time.time()
    a = np.array([pow(matrix_ciphertext[i], model_encryption_flatten[i], public_nsquare) for i in range(len(matrix_ciphertext))])
    print(f"time for multiply: {time.time() - begin_time}")
    begin_time = time.time()
    while (a.shape[0] >= 2):
        mid = a.shape[0] // 2
        mid_a = np.remainder(a[:mid] * a[mid: 2 * mid], public_nsquare)
        if a.shape[0] % 2:
            mid_a *= a[-1]
        a = mid_a
    print(f"time for add: {time.time() - begin_time}")
    return

def model_projection_paillier(idx, model_encryption_paillier, projection_matrix):
    # print(f"-------------------- {idx} ----------------")
    ans = [model_encryption_paillier[idx] * projection_matrix[idx] for idx in range(len(model_encryption_paillier))]
    return

def model_add(model_encryption_flatten, public_n):
    public_nsquare = public_n * public_n
    np.remainder(model_encryption_flatten * model_encryption_flatten, public_nsquare)
    return

def model_add_paillier(model_encryption_paillier):
    a = [model_encryption_paillier[idx] + model_encryption_paillier[idx] for idx in range(len(model_encryption_paillier))]
    return

def dict_ndarray2json(dic):
    """
    :param dic:
        {key: {ciphertext: X, exponent: X}, ...}
        type(x) == np.ndarray
    :return: str
            eval(str) = {key: {ciphertext: X, exponent: X}, ...}
            type(x) == list
    """
    json_result = defaultdict(dict)
    for key, item in dic.items():
        for keyword, keyitem in item.items():
            if type(keyitem) == np.ndarray:
                keyitem = item[keyword].tolist().copy()
            elif type(keyitem) != list:
                raise TypeError('Expected list or np.ndarray type but got: %s' % type(keyitem))
            json_result[key][keyword] = keyitem
    return json.dumps(json_result)

def dict_torch2ndarray(dic):
    dicc = dict()
    for key, val in dic.items():
        dicc[key] = val.numpy().copy()
    return dicc

def json2dict_ndarray(json_str):
    """
    :param json_str:
        "{key: {ciphertext: X, exponent: X}, ...}"
        type(x) == list
    :return:
        dic
            {key: {ciphertext: X, exponent: X}, ...}
            type(x) == np.ndarray
    """
    dict_ndarray = defaultdict(dict)
    dic_list = eval(json_str)
    for key, item in dic_list.items():
        for keyword, keyitem in item.items():
            if type(keyitem) == list:
                keyitem = np.array(item[keyword])
            elif type(keyitem) != np.ndarray:
                raise TypeError('Expected list or np.ndarray type but got: %s' % type(keyitem))
            dict_ndarray[key][keyword] = keyitem
    return dict_ndarray

def dict_comp(dict1, dict2):
    if len(dict1) != len(dict2):
        return False
    for key, item1 in dict1.items():
        if key not in dict2:
            return False
        if type(item1) == torch.Tensor:
            item_dict1 = item1.numpy().copy()
        elif type(item1) == list:
            item_dict1 = np.array(item1)
        elif type(item1) != np.ndarray:
            item_dict1 = item1
        else:
            raise TypeError('dict1: key = %s. Expected torch.Tensor, list or np.ndarray type but got: %s' % (str(key), type(item1)))
        item2 = dict2[key]
        if type(item2) == torch.Tensor:
            item_dict2 = item2.numpy().copy()
        elif type(item2) == list:
            item_dict2 = np.array(item2)
        elif type(item2) == np.ndarray:
            item_dict2 = item2
        else:
            raise TypeError('dict2: key = %s: Expected torch.Tensor, list or np.ndarray type but got: %s' % (str(key), type(item1)))
        if not np.all(item_dict1 == item_dict2):
            print(f"The comparison for {key} is False")
            return False
    return True

def dict_torch2list(dic):
    dicc = dict()
    for key, val in dic.items():
        dicc[key] = val.tolist().copy()
    return dicc

def model_size(cnn_weights, public_key):
    cnn_tran = dict()
    cnn_weights_ndarray = dict_torch2ndarray(cnn_weights)

    for key in cnn_weights.keys():
        # if key!= "conv1.weight":
        #     continue
        item_shape = cnn_weights[key].shape
        # print(key, item_shape)
        weight = cnn_weights_ndarray[key].flatten()
        weight_tran = np.ndarray(shape=weight.shape, dtype=object)
        for idx in range(len(weight)):
            item_c = public_key.encrypt(weight[idx].item())
            item_cdict = {"c.c": item_c._EncryptedNumber__ciphertext, "c.e": item_c.exponent}
            weight_tran[idx] = json.dumps(item_cdict)

        weight_tran = weight_tran.reshape(item_shape)
        # print(key, weight_tran.shape)
        cnn_tran[key] = weight_tran.tolist()  # list to json

    cnn_tranT = json.dumps(cnn_tran)
    return getsizeof(cnn_tranT) / 2 ** 20

if __name__ == "__main__":
    times = 20
    vector = 1
    connected_client_number = 1
    c = 1
    time_encryption = np.empty([times])
    time_encryption_paillier = np.empty([times])
    time_decryption = np.empty([times])
    time_projection_multiply = np.empty([times])
    time_projection_multiply_paillier = np.empty([times])
    time_projection_add_paillier = np.empty([times])
    time_projection_add = np.empty([times])
    time_semi = np.empty([times])
    time_end = np.empty([times])
    size_encryption = np.empty([times])
    size = np.empty([times])
    paillier_flag = True
    thread_number = 32 * 1

    for idx in range(times):
        if not (idx % 10):
            print(f"------------------------------------------ INDEX {idx} --------------------------------------------")
            print(time_end[idx - 10: idx])
        time_begin = time.time()
        cnn = MNIST_CNN()
        cnn_weights = cnn.state_dict()
        # print("The number of parameters: ", sum([p.data.nelement() for p in cnn.parameters()]))

        ############################  KEY GENERATION ##########################################
        public_key, private_key = paillier.generate_paillier_keypair(n_length=128)
        # ------ PUBLIC ----- #
        public_n = public_key.n
        # ------ PRIVATE ----- #
        private_p = private_key.p
        private_q = private_key.q

        # ------ Paillier ----- #
        time_en_begin = time.time()
        model_dict_encryption_paillier = model_encryption_paillier(cnn_weights, public_n)
        time_encryption_paillier[idx] = time.time() - time_en_begin
        print(f"The {idx}-th experiment: paillier encryption time-usage {time_encryption_paillier[idx]} ")
        model_encryption_paillier_flatten = []
        keys = list(model_dict_encryption_paillier.keys())
        for key in keys:
            model_encryption_paillier_flatten += list(np.array(model_dict_encryption_paillier[key]).flatten())
        # ------ porjection -----
        length_mini = len(model_encryption_paillier_flatten) // c
        model_encryption_mini = model_encryption_paillier_flatten[0: length_mini]
        matrix = list(np.random.rand(len(model_encryption_mini)))
        time_en_begin = time.time()
        threadLock = threading.Lock()
        threads = []
        threads_init = dict()

        for i in range(thread_number):
            matrix = list(np.random.rand(len(model_encryption_mini)))
            threads_init[i] = myThread(idx, matrix, model_encryption_mini)

        for i in range(thread_number):
            threads_init[i].start()

        for i in range(thread_number):
            threads.append(threads_init[i])

        # Wait for all threads to complete
        for t in threads:
            t.join()

        a = time.time() - time_en_begin
        if a < 10000:
            time_projection_multiply_paillier[idx] = (time.time() - time_en_begin)

        # time_en_begin = time.time()
        # model_projection_paillier(model_encryption_paillier_flatten, matrix)
        # time_projection_multiply_paillier[idx] += time.time() - time_en_begin
        print(f"The {idx}-th experiment: paillier projection time-usage {time_projection_multiply_paillier[idx]} ")
        time_en_begin = time.time()
        model_add_paillier(model_encryption_paillier_flatten)
        time_projection_add_paillier[idx] = time.time() - time_en_begin
        print(f"The {idx}-th experiment: paillier add time-usage {time_projection_add_paillier[idx]} ")

        model_dict_encryption = model_encryption(cnn_weights, public_n, r_value=True)
        time_de_begin = time.time()
        model_dict_decryption = model_decryption_paillier(model_dict_encryption, public_n, private_p, private_q)
        time_decryption[idx] = time.time() - time_de_begin
        comp_flag = dict_comp(cnn_weights, model_dict_decryption)
        print(f"The {idx}-th experiment: paillier decryption time-usage {time_decryption[idx]} ")

        # ------ Self-developed ----- #
        if not paillier_flag:
            time_en_begin = time.time()
            model_dict_encryption = model_encryption(cnn_weights, public_n, r_value=True)
            time_encryption[idx] = time.time() - time_en_begin
            keys = list(model_dict_encryption.keys())
            model_encryption_flatten = np.array(model_dict_encryption[keys[0]]["ciphertext"]).flatten()
            for key in keys:
                if key != keys[0]:
                    model_encryption_flatten = np.hstack(
                        (model_encryption_flatten, np.array(model_dict_encryption[key]["ciphertext"]).flatten()))
            projection_time_multiple = time.time()
            for vec in range(vector):
                matrix = np.random.rand(model_encryption_flatten.shape[0])
                model_projection(model_encryption_flatten, matrix, public_n)
            time_projection_multiply[idx] = time.time() - projection_time_multiple
            projection_time_add = time.time()
            for _ in range(connected_client_number):
                model_add(model_encryption_flatten, public_n)
            time_projection_add[idx] = time.time() - projection_time_add
            cnn_tt = json.dumps(dict_torch2list(cnn_weights))
            size[idx] = getsizeof(cnn_tt) / 2 ** 20
            size_encryption[idx] = model_size(cnn_weights, public_key)
            # model_json = dict_ndarray2json(model_dict_encryption)
            # model_dict = json2dict_ndarray(model_json)

            print(f"The comparison between {retrieve_name(cnn_weights)} and {retrieve_name(model_dict_decryption)} is {comp_flag}")
            time_end[idx] = time.time() - time_begin
            print(f"The {idx}-th experiment: projection time-usage {time_projection_multiply[idx]} ")
            print(f"The {idx}-th experiment: add time-usage {time_projection_add[idx]} ")
            print(f"The {idx}-th experiment: time-usage {time_end[idx]} \n")

    print("\n ############################################################################# \n")
    print(f"time for encryption paillier {np.average(time_encryption_paillier)}, {np.std(time_encryption_paillier)}")
    print(
        f"time for multiply paillier {np.average(time_projection_multiply_paillier)}, {np.std(time_projection_multiply_paillier)}")
    print(f"time for add paillier {np.average(time_projection_add_paillier)}, {np.std(time_projection_add_paillier)}")
    print(f"time for decryption {np.average(time_decryption)}, {np.std(time_decryption)}")
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! self-developed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print(f"time for encryption {np.average(time_encryption)}, {np.std(time_encryption)}")
    # print(f"time for projection {np.average(time_projection_multiply)}, {np.std(time_projection_multiply)}")
    # print(f"time for add {np.average(time_projection_add)}, {np.std(time_projection_add)}")
    print(f"time for the whole process {np.average(time_end)}, {np.std(time_end)}")
    # print(f"size for encryption {np.average(size_encryption)}, {np.std(size_encryption)}")
    # print(f"size  {np.average(size)}, {np.std(size)}")
    print("Heyha!")

    ans_record = dict()
    ans_record["projection"] = time_projection_multiply_paillier
    ans_record["add"] = time_projection_add_paillier
    ans_record["decryption"] = time_decryption
    ans_record["encryption"] = time_encryption_paillier
    now = datetime.datetime.now()
    formatted_time = now.strftime("%m_%d_%H_%M")
    dff = pd.DataFrame(ans_record)
    df_t = dff.T
    df_t.to_excel(formatted_time + ".xlsx", index=True)


