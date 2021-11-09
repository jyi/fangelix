import logging
import numpy as np
from custom_types import Chunk, Block, Sample, BitSeq, Proposal, \
    Ebits, EBitsSeq, BlockEbits, Cost, Location

logger = logging.getLogger('bin_utils')


class Bin:

    '''
    e.g. one_num(2) returns an int correspoding to binary 11.
    '''
    @staticmethod
    def ones(n) -> int:
        if n == 0:
            return 0
        else:
            bin_str = ''.join(map(str, np.ones(n, int)))
            return int(bin_str, 2)

    '''
    left-shift block by n bits.
    '''
    @staticmethod
    def lshift(block: Block, n: int, chunk_bits: int):
        block_copy = np.copy(block)
        for idx, chunk in enumerate(block):
            chunk_shifted = chunk << n
            mask = 0
            for degree, carry in enumerate(block[idx + 1:]):
                if n + chunk_bits - 1 < chunk_bits * (degree + 1):
                    break
                mask |= (carry << n) >> (chunk_bits * (degree + 1))
            chunk_shifted |= mask
            block_copy[idx] = chunk_shifted & Bin.ones(chunk_bits)
        return block_copy

    @staticmethod
    def rshift(block: Block, n: int, chunk_bits: int):
        def get_next_carry(v, n_in_chunk):
            mask_bits = Bin.ones(n_in_chunk)
            return (v & mask_bits) << (chunk_bits - n_in_chunk)

        chunk_shift = int(np.floor(n / chunk_bits))
        block_copy = np.copy(block[0:len(block) - chunk_shift])
        n_in_chunk = n % chunk_bits
        if n_in_chunk != 0:
            cur_carry = 0
            for idx, chunk in enumerate(block_copy):
                next_carry = get_next_carry(chunk, n_in_chunk)
                block_copy[idx] = (chunk >> n_in_chunk) | cur_carry
                cur_carry = next_carry
        num_of_shifted_chuks = len(block) - len(block_copy)
        if num_of_shifted_chuks > 0:
            block_copy = np.append(np.zeros(num_of_shifted_chuks, int), block_copy)
        return block_copy

    @staticmethod
    def bin_str(int_num: int, digits: int) -> str:
        return '{0:b}'.format(int_num).zfill(digits)

    @staticmethod
    def flip_count_between_blocks(block1, block2):
        assert len(block1) == len(block2)
        count = 0
        for i in range(len(block1)):
            count += Bin.flip_count_between_nums(block1[i], block2[i])
        return count

    '''
    Return count of bit differences betwee num1 and num2
    '''
    @staticmethod
    def flip_count_between_nums(num1, num2):
        def countSetBits(n):
            count = 0
            while n:
                count += n & 1
                n >>= 1
            return count

        # logger.debug('[flip_count_between_nums] num1: {}'.format(num1))
        # logger.debug('[flip_count_between_nums] num2: {}'.format(num2))
        rst = countSetBits(int(num1) ^ int(num2))
        return rst

    @staticmethod
    def copy_last_bits(src_block: Block, size: int, dst_block: Block, chunk_bits: int):
        dst_block_copy = np.copy(dst_block)
        whole_copy_size = size // chunk_bits
        for i in range(whole_copy_size):
            idx = len(dst_block) - 1 - i
            dst_block_copy[idx] = src_block[idx]

        rem = size % chunk_bits
        partial_copy_idx = len(dst_block) - 1 - whole_copy_size

        mask = Bin.ones(rem)
        dst_block_copy[partial_copy_idx] = (src_block[partial_copy_idx] & mask) \
                                           | (dst_block[partial_copy_idx] & ~mask)
        return dst_block_copy

    @staticmethod
    def copy_bit(src_block: Block, bit: int, dst_block: Block, chunk_bits: int):
        # logger.debug('src_block: {}'.format(src_block))
        # logger.debug('dst_block: {}'.format(dst_block))
        # logger.debug('bit: {}'.format(bit))
        # logger.debug('chunk_bits: {}'.format(chunk_bits))

        assert len(src_block) == len(dst_block)

        dst_block_copy = np.copy(dst_block)
        chunk_idx = bit // chunk_bits
        rem = bit % chunk_bits

        mask = 1 << (chunk_bits - 1 - rem)
        dst_block_copy[chunk_idx] = (src_block[chunk_idx] & mask) | (dst_block[chunk_idx] & ~mask)
        return dst_block_copy

    @staticmethod
    def pad_zeros(unpadded: Block, head_pad_len: int, tail_pad_len: int, type=str):
        # unpadded: a list of either '1' or '0'
        # logger.debug('[pad_zeros] unpadded: {}'.format(unpadded))
        padded = np.pad(unpadded,
                        (head_pad_len, tail_pad_len),
                        mode='constant', constant_values=(0))
        if len(unpadded) == 0:
            # if empty, np.pad returns an array of float 0.0
            padded = padded.astype(int).astype(type)
        # logger.debug('[pad_zeros] padded: {}'.format(padded))
        return padded

    @staticmethod
    def normalize_bit(bit_char):
        if bit_char == '0' or bit_char == '1':
            return bit_char
        else:
            return '1'
