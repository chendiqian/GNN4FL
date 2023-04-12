import numpy as np
import sys
from sys import getsizeof
import json
import fractions
import math
import random
import inspect
import time

import phe as paillier
from phe import EncodedNumber, EncryptedNumber
from phe.util import invert, powmod, mulmod, getprimeover, isqrt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mnist_cnn import MNIST_CNN

LOG2_BASE = 4.0
FLOAT_MANTISSA_BITS = sys.float_info.mant_dig
BASE = 16

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

def model_projection(model_dict_encryption, projection_matirx, public_n, private_p, private_q):
    """
        :param model_dict_encryption:
            {key: {ciphertext: X, exponent: X}, ...}
            type(x) == np.ndarray
        :return:
            projection_dict_encryption
                {key: {ciphertext: X, exponent: X}, ...}
                type(x) == np.ndarray
        """

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

if __name__ == "__main__":
    times = 1
    time_encryption = np.empty([times])
    time_decryption = np.empty([times])
    time_end = np.empty([times])

    for idx in range(times):
        if not (idx % 10):
            print(f"------------------------------------------ INDEX {idx} --------------------------------------------")
            print(time_end[idx - 10: idx])
        time_begin = time.time()
        cnn = MNIST_CNN()
        cnn_weights = cnn.state_dict()
        print("The number of parameters: ", sum([p.data.nelement() for p in cnn.parameters()]))

        ############################  KEY GENERATION ##########################################
        public_key, private_key = paillier.generate_paillier_keypair(n_length=512)
        # ------ PUBLIC ----- #
        public_n = public_key.n
        # ------ PRIVATE ----- #
        private_p = private_key.p
        private_q = private_key.q

        time_en_begin = time.time()
        model_dict_encryption = model_encryption(cnn_weights, public_n)
        time_encryption[idx] = time.time() - time_en_begin
        # model_json = dict_ndarray2json(model_dict_encryption)
        # model_dict = json2dict_ndarray(model_json)
        time_de_begin = time.time()
        model_dict_decryption = model_decryption(model_dict_encryption, public_n, private_p, private_q)
        time_decryption[idx] = time.time() - time_de_begin

        comp_flag = dict_comp(cnn_weights, model_dict_decryption)
        print(f"The comparison between {retrieve_name(cnn_weights)} and {retrieve_name(model_dict_decryption)} is {comp_flag}")
        time_end[idx] = time.time() - time_begin
        print(f"The {idx}-th experiment: time-usage {time_end[idx]} ")


    print(f"time for encryption {np.average(time_encryption)}, {np.std(time_encryption)}")
    print(f"time for decryption {np.average(time_decryption)}, {np.std(time_decryption)}")
    print(f"time for the whole process {np.average(time_end)}, {np.std(time_end)}")
    print("Heyha!")




