"""Read BrainVoyager VTC file format, using msk indeces"""

import struct
import numpy as np
from bvbabel.utils import read_variable_length_string



# =============================================================================
def read_vtc_msk(filename, msk, rearrange_data_axes=True):
    """Selectively read BrainVoyager VTC file.
    Using a mask file to sellectively search and read in only some data,
    Saving in both read in time, and memory allocation.
    
    Parameters
    ----------
    filename : string
        Path to file.
    msk : bool-ndarray, or tuple of ndarray
        accepts a 3D boolean numpy mask o
            - e.g. np.ones(img.shape[:-1])
        or a tuple of lin indeces - result of np.where(msk_bool) 
         the second approach is especially usefull when loading a vtc file in chunks
            - e.g. indeces =np.where(np.ones(img.shape[:-1]))
                   chunk_msk = tuple(arr[:500] for arr in indeces)
    rearrange_data_axes : bool
        When 'False', axes are intended to follow LIP+ terminology used
        internally in BrainVoyager:
            - 1st axis is Right to "L"eft.
            - 2nd axis is Superior to "I"nferior.
            - 3rd axis is Anterior to "P"osterior.
        When 'True' axes are intended to follow nibabel RAS+ terminology:
            - 1st axis is Left to "R"ight.
            - 2nd axis is Posterior to "A"nterior.
            - 3rd axis is Inferior to "S"uperior.
    Returns
    -------
    header : dictionary
        Pre-data and post-data headers.
    data : 3D numpy.array
        Image data.
    """
    
    # check if input msk is boolean(like) 
    #  else put into indeces array
    if len(msk)>3:
        msk = np.where(msk)

    # put msk indexes in array
    msk_array = np.array([x for x in msk]).transpose()
        
    # check the msk length
    mskleng = msk_array.shape[0]

    header = dict()
    with open(filename, 'rb') as f:
        # Expected binary data: short int (2 bytes)
        data, = struct.unpack('<h', f.read(2))
        header["File version"] = data

        # Expected binary data: variable-length string
        data = read_variable_length_string(f)
        header["Source FMR name"] = data

        # Expected binary data: short int (2 bytes)
        data, = struct.unpack('<h', f.read(2))
        header["Protocol attached"] = data

        if header["Protocol attached"] > 0:
            # Expected binary data: variable-length string
            data = read_variable_length_string(f)
            header["Protocol name"] = data
        else:
            header["Protocol name"] = ""

        # Expected binary data: short int (2 bytes)
        data, = struct.unpack('<h', f.read(2))
        header["Current protocol index"] = data
        data, = struct.unpack('<h', f.read(2))
        header["Data type (1:short int, 2:float)"] = data
        data, = struct.unpack('<h', f.read(2))
        header["Nr time points"] = data
        data, = struct.unpack('<h', f.read(2))
        header["VTC resolution relative to VMR (1, 2, or 3)"] = data

        data, = struct.unpack('<h', f.read(2))
        header["XStart"] = data
        data, = struct.unpack('<h', f.read(2))
        header["XEnd"] = data
        data, = struct.unpack('<h', f.read(2))
        header["YStart"] = data
        data, = struct.unpack('<h', f.read(2))
        header["YEnd"] = data
        data, = struct.unpack('<h', f.read(2))
        header["ZStart"] = data
        data, = struct.unpack('<h', f.read(2))
        header["ZEnd"] = data

        # Expected binary data: char (1 byte)
        data, = struct.unpack('<B', f.read(1))
        header["L-R convention (0:unknown, 1:radiological, 2:neurological)"] = data
        data, = struct.unpack('<B', f.read(1))
        header["Reference space (0:unknown, 1:native, 2:ACPC, 3:Tal, 4:MNI)"] = data

        # Expected binary data: char (4 bytes)
        data, = struct.unpack('<f', f.read(4))
        header["TR (ms)"] = data

        # Prepare dimensions of VTC data array
        VTC_resolution = header["VTC resolution relative to VMR (1, 2, or 3)"]
        DimX = (header["XEnd"] - header["XStart"]) // VTC_resolution
        DimY = (header["YEnd"] - header["YStart"]) // VTC_resolution
        DimZ = (header["ZEnd"] - header["ZStart"]) // VTC_resolution
        DimT = header["Nr time points"]
        
        # ---------------------------------------------------------------------
        # Data Characteristics
        # ---------------------------------------------------------------------
        # Each data element (intensity value) is
        # represented in 2 bytes (unsigned short) or 4 bytes (float) as
        # specified in "data type" entry.
        
        if header["Data type (1:short int, 2:float)"] == 1:
            bit_len = 2
            dtype = '<h'
        elif header["Data type (1:short int, 2:float)"] == 2:
            bit_len = 4
            dtype = '<f'
        else:
            raise("Unrecognized VTC data_img type.")

        # ---------------------------------------------------------------------
        # Calculate Positions of data within binary data
        # ---------------------------------------------------------------------
        # Note: This altered version of bvbabel's read vtc:
        #  first calculates what data to load, and its location within the
        #  binary file. 
        # 
        # The (full) data is organized in four loops (DimZ,DimY,DimX,DimY).
        #  to minimize the amount of chunks to load, the script automatically calculates 
        #  neighboring binary data, prunes the amount of chucks

        # calculate the offset multiplyer of the nested loop
        offset_array = np.array([DimY * DimX * DimT, # matches 1st dim of msk file
                                 DimT,               # matches 2nd dim of msk file
                                 DimX * DimT])       # matches 3th dim of msk file

        # calculate the positional ofset of a nested loop, 
        #  take the sum to get the absolute position
        msk_array = msk_array * offset_array
        msk_array = msk_array.sum(axis=1)
        
        # inverse reading direction of binary data to match msk
        #  if true(default) count indexes back to front
        if rearrange_data_axes is True:
            msk_array = (DimZ * DimY * DimX * DimT) - (msk_array + DimT)
            
        # sort the indeces, since we are reading sequenctially in 'f'
        #  also save the reverse indexes, to go back to original space more easily
        sorted_idx = np.argsort(msk_array,kind='stable')
        rev_idx = np.empty(len(sorted_idx), dtype='int')
        rev_idx[sorted_idx] = np.arange(len(sorted_idx))
        # apply sorted indexes
        msk_array = msk_array[sorted_idx]
        
        # calulate where the subsequent binary data directly neibors 
        #  Taking DimT as the 'minimum' reading length
        # then split the non-neigbhoring data - e.g. [[1,2],[4],[8,9]]
        split_points = np.where(np.ediff1d(msk_array) != DimT)[0] +1
        msk_group = np.split(msk_array, split_points)

        # calculate the reading length by multiplying grouplength with DimT
        #  aditionally prune the msk_array by only taking first times of the
        #  nested array
        readlen = np.array([len(g) for g in msk_group]) * DimT
        msk_array = np.array([g[0] for g in msk_group])

        # make the mask array relative instead of absolute [2,3,6] > [2,1,3]
        msk_array[1:] = msk_array[1:] - (msk_array + readlen)[:-1]

        # multiply the mask array with the length of a datapoint
        msk_array *= bit_len
        
        # do some intermediate cleanup of large arrays
        del split_points
        del msk_group

        # ---------------------------------------------------------------------
        # Read VTC data
        # ---------------------------------------------------------------------
        # NOTE: since we are sequentially loading chunks of vtc data,
        #  the outputted file will be the same dimension as the inputted
        #  indeces.
        # returned file will have the schape of [len(msk_indeces), DimT]
        
        # predefine raw data array
        data_img = np.zeros(mskleng * DimT)
        # define arrays of start and end indeces for 'data_img'
        img_idx = np.vstack((np.cumsum(readlen) - readlen,
                             np.cumsum(readlen)))
        
        # loop over loading chunks 
        for chunk in range(len(msk_array)):
            # load chunk of data and place in data_img
            data_img[img_idx[0,chunk] : 
                     img_idx[1,chunk]] = np.fromfile(f, dtype=dtype, 
                                                      count=readlen[chunk], 
                                                      sep="",
                                                      offset=msk_array[chunk])
           
    # rearange shape to be [voxels * time], and reverse previous sorting
    data_img = np.reshape(data_img, (mskleng, DimT))
    
    # lastly we revert previous msk-style > bv-style sorting
    data_img = data_img[rev_idx,:]

    return header, data_img


# =============================================================================
def get_vtc_dims(header):
    """input header file and return original vtc dimensions"""
    
    # get dimensions of VTC data array
    VTC_resolution = header["VTC resolution relative to VMR (1, 2, or 3)"]
    DimX = (header["XEnd"] - header["XStart"]) // VTC_resolution
    DimY = (header["YEnd"] - header["YStart"]) // VTC_resolution
    DimZ = (header["ZEnd"] - header["ZStart"]) // VTC_resolution
    DimT = header["Nr time points"]
    
    return tuple((DimZ, DimX, DimY, DimT))


# =============================================================================
def chunk_msk(msk, chunksize):
    """Chunk BrainVoyager mask file based on desired chucksize. 
    then split the mask indeces into the desired chucks
    
    
    Parameters
    ----------
    msk : tuple
        Masking file in tuple of indeces (np.where(msk) of boolean)
    chunksize : int
        Number of values to include within a single chunk
        
    Returns
    -------
    msk_group : list of tuples
        list of tuples where each tuple =< chucksize
    """
    
    # calculate the full length
    full_len = msk[0].shape[0]

    # based on full length and chucksize, group linear indeces
    split_points = np.where(np.arange(0, full_len) % chunksize == 0)[0][1:]
    msk_group = [np.split(m, split_points) for m in msk]

    # rearange format to make easier to use
    msk_group = [tuple((curmsk[chuck] for curmsk in msk_group)) for chuck in range(len(msk_group[0]))]
    
    return(msk_group)