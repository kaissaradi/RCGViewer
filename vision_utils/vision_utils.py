import numpy as np
import struct
import os
# from progressbar import *
from tqdm import tqdm
import bin2py
import vision_utils.vision_util_cpp_extensions as utilcpp


RW_BLOCKSIZE = 2000000
TTL_THRESHOLD = -1000
TTL_CHANNEL = 0

N_BYTES_BOOLEAN = 1
N_BYTES_16BIT = 2
N_BYTES_32BIT = 4
N_BYTES_64BIT = 8

def get_litke_triggers(bin_path):
    epoch_starts = []
    epoch_ends = []
    with bin2py.PyBinFileReader(bin_path, chunk_samples=RW_BLOCKSIZE, is_row_major=True) as pbfr:
        for start_idx in range(0, pbfr.length, RW_BLOCKSIZE):
            n_samples_to_get = min(RW_BLOCKSIZE, pbfr.length - start_idx)
            samples = pbfr.get_data_for_electrode(TTL_CHANNEL, start_idx, n_samples_to_get)
            # Find the threshold crossings at the beginning and end of each epoch.
            below_threshold = (samples < TTL_THRESHOLD)
            above_threshold = np.logical_not(below_threshold)
            # Epoch starts.
            above_to_below_threshold = np.logical_and.reduce([
                above_threshold[:-1],
                below_threshold[1:]
            ])
            trigger_indices = np.argwhere(above_to_below_threshold) + start_idx
            epoch_starts.append(trigger_indices[:, 0])
            below_to_above_threshold = np.logical_and.reduce([
                below_threshold[:-1],
                above_threshold[1:]
            ])
            trigger_indices = np.argwhere(below_to_above_threshold) + start_idx
            epoch_ends.append(trigger_indices[:, 0])
    epoch_starts = np.concatenate(epoch_starts, axis=0)
    epoch_ends = np.concatenate(epoch_ends, axis=0)
    return epoch_starts, epoch_ends

# Quality control methods.
def find_refractory_violations(isi, refractory_threshold=0.1, isi_binning=0.5):
    """
    Determine the cluster indices that violate refractorieness (<2ms isi) 
    Parameters:
        acf: interspike interval distribution (sum=1)
        refractory_threshold: Threshold/tolerance for refractory violations (default: 0.2%)
        isi_binning: bin width of isi distribution in msec (default: 0.5)
    """
    pct_refractory = compute_refractory_pct(isi, isi_binning=isi_binning)
    idx_bad = np.argwhere((pct_refractory > refractory_threshold))[:,0]
    idx_good = np.argwhere((pct_refractory <= refractory_threshold))[:,0]
    return idx_bad, idx_good

def compute_refractory_pct(isi, isi_binning=0.5):
    """
    Calculate the percentage of spikes that violate refractoriness (<2ms isi) 
    Parameters:
        isi: interspike interval distribution (sum=1)
        isi_binning: bin width of isi distribution in msec (default: 0.5)
    """
    num_bins = np.floor(1.5 / isi_binning).astype(int)
    return np.sum(isi[:,:num_bins], axis=1) * 100

def compute_snr(timecourse_matrix):
    """
    Calculate the signal-to-noise ratio for the time courses.
    Parameters:
        timecourse_matrix: Matrix of time courses
    """
    half_time = np.floor(timecourse_matrix.shape[1]/2).astype(int) + 1
    np.seterr(all='ignore')
    snr = np.divide(np.std(timecourse_matrix[:,1:half_time,(0,2)], axis=1), np.std(timecourse_matrix[:,half_time:,(0,2)], axis=1))
    snr[np.isnan(snr)] = 0.0 # Set NaN's to zero.
    snr = np.nanmax(snr, axis=1)
    return snr

def find_bad_cells(timecourse_matrix, isi, total_spikes, refractory_threshold=0.1, snr_threshold=2.0, isi_binning=0.5, count_threshold=1000):
    """! Determine the cluster indices that violate basic quality control criteria.
    Parameters:
        @param timecourse_matrix: Matrix of time courses
        @param acf: interspike interval distribution (sum=1)
        @param total_spikes: Total number of spikes for each cell
        @param refractory_threshold: Threshold/tolerance for refractory violations (default: 0.2%)
        @param snr_threshold: Threshold/tolerance for signal-to-noise ratio (default: 2.0)
        @param isi_binning: bin width of isi distribution in msec (default: 0.5)
        @param count_threshold: Minimum number of spikes required to be considered a good cell (default: 1000)
    Returns:
        @return idx_bad: Indices of bad cells
        @return idx_good: Indices of good cells
    """
    # Calculate SNR.
    snr = compute_snr(timecourse_matrix)
    # Calculate the percentage of spikes that violate refractoriness (<2ms isi)
    pct_refractory = compute_refractory_pct(isi, isi_binning)
    # May have to play with the cutoff
    idx_bad = np.argwhere((snr < snr_threshold) | (pct_refractory > refractory_threshold) | (total_spikes < count_threshold))[:,0]
    idx_good = np.argwhere((snr >= snr_threshold) & (pct_refractory <= refractory_threshold) & (total_spikes >= count_threshold))[:,0]
    return idx_bad, idx_good

class STAWriter(object):
    HEADER_LENGTH_BYTES = (7 * N_BYTES_32BIT) + N_BYTES_64BIT + 128
    HEADER_PAD_LENGTH_BYTES = N_BYTES_64BIT + 128

    def __init__(self, filepath):
        """
        Constructor method
        Parameters:
            filepath: Full path to the .sta file that you intend to write
        """
        self.filepath = filepath
        self.version = 32
        self.max_neurons = 10000
        self.fp = None

    def write(self, sta: np.ndarray=None, ste: np.ndarray=None, cluster_id: np.ndarray=None, stixel_size: float=30.0, frame_refresh: float=1000/120):
        """
        Arguments:
            sta: Tensor containing the spike-triggered average. Dimensions:[cell,time,y,x,RGB]  (type: numpy.ndarray)
            cluster_id: Array containing the cluster/cell ids. Dimensions:[cell] (type: numpy.ndarray)
            stixel_size: Scalar value for the edge length of each stixel in microns (type: float)
            frame_refresh: Scalar value for the temporal refresh of the STA in milliseconds (type: float)
        """
        n_cells = sta.shape[0] # Number of cells
        sta_width = sta.shape[3] # Width
        sta_height = sta.shape[2] # Height
        sta_depth = sta.shape[1] # Time depth
        sta_colors = sta.shape[4] # Number of color channels for STA

        if ste is None:
            ste = np.zeros_like(sta)

        # Reverse the STA time dimension for vision.
        sta = sta[:,::-1,:,:,:]

        # Rescale the STA from contrast to 0-1 for Vision, if needed.
        if np.min(sta) > -0.001:
            sta = 0.5 * sta + 0.5

        # Calculate the size in bytes of a single cell's STA.
        sta_size = 12 + sta_depth * ((sta_width * sta_height * sta_colors + 2) * 8)

        # Get the first cell's location.
        first_location = n_cells * 4 + n_cells * 8 + 164
        
        # Open the file for writing.
        with open(self.filepath, 'wb') as fp:
            # Write the header
            fp.write( struct.pack('>IIIII', self.version, n_cells, sta_width, sta_height, sta_depth) )

            # Write the stixel size and frame refresh to the header.
            fp.write( struct.pack('>d', stixel_size) )
            fp.write( struct.pack('>d', frame_refresh) )

            # Skip the header (164 bytes) and write the cluster_ids and data locations.
            fp.seek(STAWriter.HEADER_LENGTH_BYTES, os.SEEK_SET) # fp.seek(164, os.SEEK_SET)

            # Write the Cell Ids and data locations
            for i, id in enumerate(cluster_id):
                fp.write( struct.pack('>I', id) ) # Cluster ID or Cell Number
                fp.write( struct.pack('>Q', sta_size * i + first_location) ) # Data locations

            # Write each STA.
            for cell in tqdm(range(n_cells), desc ="Writing STA file ..."):
                # Write frame refresh and depth for each STA.
                fp.write( struct.pack('>d', frame_refresh) )
                fp.write( struct.pack('>I', sta_depth) )
                output_buffer = utilcpp.pack_sta_buffer(sta[cell,:,:,:,0],ste[cell,:,:,:,0],sta[cell,:,:,:,1],ste[cell,:,:,:,1],sta[cell,:,:,:,2],ste[cell,:,:,:,2], stixel_size)
                fp.write(output_buffer)
                # for t in range(sta_depth):
                    # Write the width, height, and stixel size for each frame.
                    # fp.write( struct.pack('>II', sta_width, sta_height) )
                    # fp.write( struct.pack('>d', stixel_size) )

                    # output_buffer = utilcpp.pack_sta_buffer(sta[cell,t,:,:,0],ste[cell,t,:,:,0],sta[cell,t,:,:,1],ste[cell,t,:,:,1],sta[cell,t,:,:,2],ste[cell,t,:,:,2], stixel_size)
                    # fp.write(output_buffer)
                    
                    # for y in range(sta_height):
                    #     for x in range(sta_width):
                    #         for color in range(sta_colors):
                    #             fp.write( struct.pack('>f', sta[cell,t,y,x,color]) ) # STA value
                    #             if ste is None:
                    #                 fp.write( struct.pack('>f', 0.0) ) # Error value
                    #             else:
                    #                 fp.write( struct.pack('>f', ste[cell,t,y,x,color]) ) # STE/STV value
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fp is not None:
            self.fp.close()

    def close(self):
        if self.fp is not None:
            self.fp.close()

class ParamsWriter(object):
    PARAM_NAMES_AND_TYPES = {
        'ID': 'Double', 'classID': 'String', 'x0': 'Double', 'y0': 'Double', 'SigmaX': 'Double', 'SigmaY': 'Double', 
        'Theta': 'Double', 'gAmp': 'Double', 'contourX': 'DoubleArray', 'contourY': 'DoubleArray', 'contourArea': 'Double', 
        'simpleContourX': 'DoubleArray', 'simpleContourY': 'DoubleArray', 'simpleContourArea': 'Double', 'EIx0': 'Double', 
        'EIy0': 'Double', 'EISigmaX': 'Double', 'EISigmaY': 'Double', 'EITheta': 'Double', 'EIgAmp': 'Double', 
        'RedTimeCourse': 'DoubleArray', 'GreenTimeCourse': 'DoubleArray', 'BlueTimeCourse': 'DoubleArray', 't1': 'Double', 
        't2': 'Double', 't3': 'Double', 'a1': 'Double', 'a2': 'Double', 'a3': 'Double', 'n1': 'Double', 'n2': 'Double', 
        'n3': 'Double', 'tOffset': 'Double', 'dot': 'Double', 'dot2': 'Double', 'srm': 'Double', 'rl': 'Double', 
        'amp1': 'Double', 'amp2': 'Double', 'amp3': 'Double', 'blueness': 'Double', 'RedVTimeCourse': 'DoubleArray', 
        'GreenVTimeCourse': 'DoubleArray', 'BlueVTimeCourse': 'DoubleArray', 't1V': 'Double', 't2V': 'Double', 
        't3V': 'Double', 'a1V': 'Double', 'a2V': 'Double', 'a3V': 'Double', 'n1V': 'Double', 'n2V': 'Double', 
        'n3V': 'Double', 'tVOffset': 'Double', 'dotV': 'Double', 'dot2V': 'Double', 'srmV': 'Double', 'rlV': 'Double', 
        'amp1V': 'Double', 'amp2V': 'Double', 'amp3V': 'Double', 'bluenessV': 'Double', 'Auto': 'DoubleArray', 
        'acfBinning': 'Double', 'nSpikes': 'Double', 'acfMean': 'Double', 'acfRMS': 'Double', 'contamination': 'Double',
        'isiBinning': 'Double', 'isiMean': 'Double', 'isiRMS': 'Double'}
    DOUBLE_SIZE = 10
    STRING_SIZE = 9
    
    def __init__(self, filepath: str, cluster_id: np.ndarray):
        """
        Constructor method
        Parameters:
            filepath: Full path to the .params file that you intend to write
            cluster_id: List of cluster ids that align with those contained in the .neurons file
        """

        self.filepath = filepath
        self.cluster_id = cluster_id
        # Parameter names and data types.
        self.param_names_and_types = self.getParamDict()
        self.data_sizes = [] # [10, 9, 10, 10, 10, 10, 10, 10, 6, 6, 10, 6, 6, 10, 408, 408, 408, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 6, 6, 6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1608, 10, 10, 10, 10]
        self.fp = open(filepath, 'wb')

    def check_timecourse(self, time_course: np.ndarray):
        if time_course is not None:
            num_timecourse = time_course.shape[1]
        else:
            num_timecourse = 0
        if num_timecourse == 0:
            time_course = None
        return time_course, num_timecourse

    def check_contour_data(self, contour_data: np.ndarray):
        if contour_data is not None:
            contour_counts = np.count_nonzero(contour_data[:,:,0], axis=1)
            num_contours = np.max(contour_counts)
        else:   
            num_contours = 0
        if num_contours == 0:
            contour_data = None
        return contour_data, num_contours

    def write_header(self, num_params: int, num_cells: int, maxNeurons: int=10000):
        """ Write the header of the .params file. 
        Arguments:
            num_params: Number of parameters.
            num_cells: Number of cells.
            maxNeurons: Maximum number of neurons.
        """
        # Write the header
        print('Writing file header.')
        self.fp.write( struct.pack('>III', num_params, num_cells, maxNeurons) )
    
    def write_parameter_dict(self):
        """ Write the parameter names and data types."""
        for param in self.param_names_and_types:
            for i in range(3):
                self.fp.write(struct.pack('>b', 0))
            l = len(param)
            self.fp.write(struct.pack('>b', l))
            for k in range(l):
                self.fp.write(struct.pack('>s', bytes(param[k], 'utf-8')))
            for i in range(3):
                self.fp.write(struct.pack('>b', 0))
            value = self.param_names_and_types[param]
            l = len(value)
            self.fp.write(struct.pack('>b', l))
            for k in range(l):
                self.fp.write(struct.pack('>s', bytes(value[k], 'utf-8')))
    
    def write_file_locations(self, locs: list):
        """ Write the file locations for each parameter. 
        Arguments:
            locs: List of file locations for each parameter.
        """
        for loc in locs:
            self.fp.write(struct.pack('>I', loc-2))

        end_of_fid = locs[-1]+8
        num = end_of_fid - self.fp.tell()
        for i in range(num):
            self.fp.write(struct.pack('>b', 0))
    
    def write(self, timecourse_matrix, isi, spike_count, x0, y0, sigma_x, sigma_y, 
              theta: np.ndarray=None, isi_binning: float=0.5, 
              contour_xy: np.ndarray=None, contour_area: np.ndarray=None, simple_contour_xy: np.ndarray=None, 
              simple_contour_area: np.ndarray=None, timecourse_variance: np.ndarray=None, ei_parameters: np.ndarray=None):
        """
        Arguments:
            timecourse_matrix: Timecourse matrix/temporal filters. Dimensions:[cell,time,RGB]  (type: numpy.ndarray)
            isi: Interspike interval distributions. Dimensions:[cell, time] (type: numpy.ndarray)
            spike_count: Total spike counts to noise stimulus. Dimensions:[cell] (type: numpy.ndarray)
            x0: X-location (stixels) of RF center (type: numpy.ndarray)
            y0: Y-location (stixels) of RF center (type: numpy.ndarray)
            sigma_x: X-standard deviation of RF Gaussian (type: numpy.ndarray)
            sigma_y: Y-standard deviation of RF Gaussian (type: numpy.ndarray)
            theta: Rotational angle of Gaussian RF (in degrees) (type: numpy.ndarray)
            isi_binning: Bin width (in milliseconds) used for computing the ISI distribution (type: float, default: 0.5)
            hull_vertices: Vertices of the convex hull of the RF (type: numpy.ndarray)
            hull_area: Area of the convex hull of the RF (type: numpy.ndarray)
        """
        
        num_cells = timecourse_matrix.shape[0]
        maxNeurons = 10000
        cellNameString = [0, 0, 0, 3, 65, 108, 108]

        # Set the data sizes based on the inputs.
        num_timecourse = timecourse_matrix.shape[1]
        num_isi = isi.shape[1]
        # Check inputs to make sure they are valid.
        timecourse_variance, num_v_timecourse = self.check_timecourse(timecourse_variance)
        contour_xy, num_contours = self.check_contour_data(contour_xy)
        simple_contour_xy, num_simple_contours = self.check_contour_data(simple_contour_xy)
        # Remove the irrelevant params and get the data lengths.
        self.set_data_sizes(num_timecourse=num_timecourse, num_isi=num_isi, num_v_timecourse=num_v_timecourse, num_contours=num_contours, num_simple_contours=num_simple_contours)
        
        # Get the number of relevant parameters.
        num_params = len(self.param_names_and_types)

        # Write the header
        self.write_header(num_params=num_params, num_cells=num_cells, maxNeurons=maxNeurons)

        # Write the parameter names and data types.
        self.write_parameter_dict()

        # Get the data locations.
        locs = self.get_data_locations(num_params, num_cells)

        # Write the locations.
        self.write_file_locations(locs)

        # Write the parameter values for each cell.
        param_count = 0
        for cellID in tqdm(range(num_cells), desc='Writing parameters for each cell'):
            for i in range(num_params):
                self.fp.seek(locs[param_count]-2, os.SEEK_SET)
                param_count += 1
                param_key = list(self.param_names_and_types.keys())[i]
                param_value = list(self.param_names_and_types.values())[i]
                if param_value == 'String':
                    self.fp.write(struct.pack('>h', 327))
                    for v in cellNameString:
                        self.fp.write(struct.pack('>b', v))
                elif list(self.param_names_and_types.values())[i] == 'DoubleArray':
                    is_array = False
                    if (param_key == 'RedTimeCourse' or param_key == 'GreenTimeCourse' or param_key == 'BlueTimeCourse'):
                        self.fp.write(struct.pack('>h', 319))
                        self.fp.write(struct.pack('>h', 404))
                        self.fp.write(struct.pack('>h', 0))
                        self.fp.write(struct.pack('>h', num_timecourse))
                        is_array = True
                    elif param_key == 'Auto':
                        self.fp.write(struct.pack('>h', 319))
                        self.fp.write(struct.pack('>h', 1604))
                        self.fp.write(struct.pack('>h', 0))
                        self.fp.write(struct.pack('>h', num_isi))
                        is_array = True
                    elif (timecourse_variance is not None) and (param_key == 'RedVTimeCourse' or param_key == 'GreenVTimeCourse' or param_key == 'BlueVTimeCourse'):
                        self.fp.write(struct.pack('>h', 319))
                        self.fp.write(struct.pack('>h', 404))
                        self.fp.write(struct.pack('>h', 0))
                        self.fp.write(struct.pack('>h', num_v_timecourse))
                        is_array = True
                    elif (contour_xy is not None) and (param_key == 'contourX' or param_key == 'contourY'):
                        self.fp.write(struct.pack('>h', 319))
                        self.fp.write(struct.pack('>h', 404))
                        self.fp.write(struct.pack('>h', 0))
                        self.fp.write(struct.pack('>h', num_contours))
                        is_array = True
                    elif (simple_contour_xy is not None) and (param_key == 'simpleContourX' or param_key == 'simpleContourY'):
                        self.fp.write(struct.pack('>h', 319))
                        self.fp.write(struct.pack('>h', 404))
                        self.fp.write(struct.pack('>h', 0))
                        self.fp.write(struct.pack('>h', num_simple_contours))
                        is_array = True
                    else:
                        self.fp.write(struct.pack('>h', 260))
                        self.fp.write(struct.pack('>h', 0))
                        self.fp.write(struct.pack('>h', 0))
                    
                    if is_array:
                        if param_key == 'RedTimeCourse':
                            my_array = timecourse_matrix[cellID, :, 0]
                        elif param_key == 'GreenTimeCourse':
                            my_array = timecourse_matrix[cellID, :, 1]
                        elif param_key == 'BlueTimeCourse':
                            my_array = timecourse_matrix[cellID, :, 2]
                        elif param_key == 'RedVTimeCourse':
                            my_array = timecourse_variance[cellID, :, 0]
                        elif param_key == 'GreenVTimeCourse':
                            my_array = timecourse_variance[cellID, :, 1]
                        elif param_key == 'BlueVTimeCourse':
                            my_array = timecourse_variance[cellID, :, 2]
                        elif param_key == 'contourX':
                            my_array = contour_xy[cellID, :num_contours, 0]
                        elif param_key == 'contourY':
                            my_array = contour_xy[cellID, :num_contours, 1]
                        elif param_key == 'simpleContourX':
                            my_array = simple_contour_xy[cellID, :num_simple_contours, 0]
                        elif param_key == 'simpleContourY':
                            my_array = simple_contour_xy[cellID, :num_simple_contours, 1]
                        else:
                            my_array = isi[cellID, :]
                        for val in my_array:
                            self.fp.write(struct.pack('>d', val))
                else:
                    if param_key == 'ID':
                        k = self.cluster_id[cellID] #cellID + 1
                    elif param_key == 'x0':
                        k = np.max([1.0, x0[cellID]])
                    elif param_key == 'y0':
                        k = np.max([1.0, y0[cellID]])
                    elif param_key == 'SigmaX':
                        k = np.max([1.0, sigma_x[cellID]])
                    elif param_key == 'SigmaY':
                        k = np.max([1.0, sigma_y[cellID]])
                    elif param_key == 'Theta':
                        if theta is None:
                            k = 0.0
                        else:
                            k = theta[cellID]
                    elif param_key == 'contourArea':
                        if contour_area is not None:
                            k = contour_area[cellID]
                        else:
                            k = 0.0
                    elif param_key == 'simpleContourArea':
                        if simple_contour_area is not None:
                            k = simple_contour_area[cellID]
                        else:
                            k = 0.0
                    elif param_key == 'nSpikes':
                        k = spike_count[cellID]
                    elif param_key == 'tOffset':
                        k = 0.0
                    elif (param_key == 'acfBinning' or param_key == 'isiBinning'):
                        k = isi_binning
                    elif (param_key == 'acfMean' or param_key == 'isiMean'):
                        k = np.mean(isi[cellID, :])
                    elif (param_key == 'acfRMS' or param_key == 'isiRMS'):
                        k = 43.1999
                    elif param_key == 'EIgAmp':
                        if ei_parameters is not None:
                            k = ei_parameters[cellID, 0]
                        else:
                            k = 0.0
                    elif (param_key == 'EIx0'):
                        if ei_parameters is not None:
                            k = ei_parameters[cellID, 1]
                        else:
                            k = 0.0
                    elif (param_key == 'EIy0'):
                        if ei_parameters is not None:
                            k = ei_parameters[cellID, 2]
                        else:
                            k = 0.0
                    elif (param_key == 'EISigmaX'):
                        if ei_parameters is not None:
                            k = ei_parameters[cellID, 3]
                        else:
                            k = 0.0
                    elif (param_key == 'EISigmaY'):
                        if ei_parameters is not None:
                            k = ei_parameters[cellID, 4]
                        else:
                            k = 0.0
                    elif (param_key == 'EITheta'):
                        if ei_parameters is not None:
                            k = ei_parameters[cellID, 5]
                        else:
                            k = 0.0
                    elif (param_key == 'blueness' or param_key == 'amp1' or param_key == 'amp2' or param_key == 'amp3'):
                        k = 0.0
                    else:
                        k = 0.0
                    
                    self.fp.write(struct.pack('>h', 200))
                    self.fp.write(struct.pack('>d', k))
    
    def getParamDict(self):
        # param_keys =  ['ID', 'classID', 'x0', 'y0', 'SigmaX', 'SigmaY', 'Theta', 'gAmp', 'contourX', 'contourY', 'contourArea', 'simpleContourX', 'simpleContourY', 'simpleContourArea', 'RedTimeCourse', 'GreenTimeCourse', 'BlueTimeCourse', 't1', 't2', 't3', 'a1', 'a2', 'a3', 'n1', 'n2', 'n3', 'tOffset', 'dot', 'dot2', 'srm', 'rl', 'amp1', 'amp2', 'amp3', 'blueness', 'RedVTimeCourse', 'GreenVTimeCourse', 'BlueVTimeCourse', 't1V', 't2V', 't3V', 'a1V', 'a2V', 'a3V', 'n1V', 'n2V', 'n3V', 'tVOffset', 'dotV', 'dot2V', 'srmV', 'rlV', 'amp1V', 'amp2V', 'amp3V', 'bluenessV', 'Auto', 'isiBinning', 'nSpikes', 'isiMean', 'isiRMS']
        # param_values = ['Double', 'String', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'DoubleArray', 'DoubleArray', 'Double', 'DoubleArray', 'DoubleArray', 'Double', 'DoubleArray', 'DoubleArray', 'DoubleArray', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'DoubleArray', 'DoubleArray', 'DoubleArray', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'DoubleArray', 'Double', 'Double', 'Double', 'Double']

        # paramNames = dict()
        # for i, key in enumerate(param_keys):
        #     paramNames[key] = param_values[i]
        paramNames = ParamsWriter.PARAM_NAMES_AND_TYPES.copy()
        return paramNames

    def get_data_locations(self, num_params: int, num_cells: int):
        """ Get the data locations for each parameter. 
        Arguments:
            num_params: Number of parameters.
            num_cells: Number of cells. 
        Returns:
            locs: List of data locations for each parameter.
        """
        locs = []
        locs.append(2441282)
        count = 0
        for cell in range(num_cells):
            for param in range(num_params):
                locs.append(locs[count] + self.data_sizes[param])
                count += 1
        locs = locs[:-1]
        return locs
    
    def set_data_sizes(self, num_timecourse: int=61, num_isi: int=300, num_v_timecourse: int=0, num_contours: int=0, num_simple_contours: int=0):
        """ Set the data sizes for each parameter. 
        Arguments:
            num_timecourse: Number of timecourse points.
            num_isi: Number of isi points.
            num_v_timecourse: Number of variance timecourse points.
            num_contours: Number of contour points.
            num_simple_contours: Number of simple contour points.
        """
        self.data_sizes = list()

        for key, value in self.param_names_and_types.items():
            if value == 'Double':
                self.data_sizes.append(ParamsWriter.DOUBLE_SIZE)
            elif value == 'String':
                self.data_sizes.append(ParamsWriter.STRING_SIZE)
            elif value == 'DoubleArray':
                if (key == 'RedTimeCourse' or key == 'GreenTimeCourse' or key == 'BlueTimeCourse'):
                    self.data_sizes.append((num_timecourse+1) * N_BYTES_64BIT)
                elif key == 'Auto':
                    self.data_sizes.append((num_isi+1) * N_BYTES_64BIT)
                elif (num_v_timecourse > 0) and (key == 'RedVTimeCourse' or key == 'GreenVTimeCourse' or key == 'BlueVTimeCourse'):
                    self.data_sizes.append((num_v_timecourse+1) * N_BYTES_64BIT)
                elif (num_contours > 0) and (key == 'contourX' or key == 'contourY'):
                    self.data_sizes.append((num_contours+1) * N_BYTES_64BIT)
                elif (num_simple_contours > 0) and (key == 'simpleContourX' or key == 'simpleContourY'):
                    self.data_sizes.append((num_simple_contours+1) * N_BYTES_64BIT)
                else:
                    self.data_sizes.append(6)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fp is not None:
            self.fp.close()

    def close(self):
        if self.fp is not None:
            self.fp.close()

# PARAM_FIELDS = {
#     'ID': 'Double', 'classID': 'String', 'x0': 'Double', 'y0': 'Double', 'SigmaX': 'Double', 'SigmaY': 'Double', 
#     'Theta': 'Double', 'gAmp': 'Double', 'contourX': 'DoubleArray', 'contourY': 'DoubleArray', 'contourArea': 'Double', 
#     'simpleContourX': 'DoubleArray', 'simpleContourY': 'DoubleArray', 'simpleContourArea': 'Double', 'EIx0': 'Double', 
#     'EIy0': 'Double', 'EISigmaX': 'Double', 'EISigmaY': 'Double', 'EITheta': 'Double', 'EIgAmp': 'Double', 
#     'RedTimeCourse': 'DoubleArray', 'GreenTimeCourse': 'DoubleArray', 'BlueTimeCourse': 'DoubleArray', 't1': 'Double', 
#     't2': 'Double', 't3': 'Double', 'a1': 'Double', 'a2': 'Double', 'a3': 'Double', 'n1': 'Double', 'n2': 'Double', 
#     'n3': 'Double', 'tOffset': 'Double', 'dot': 'Double', 'dot2': 'Double', 'srm': 'Double', 'rl': 'Double', 
#     'amp1': 'Double', 'amp2': 'Double', 'amp3': 'Double', 'blueness': 'Double', 'RedVTimeCourse': 'DoubleArray', 
#     'GreenVTimeCourse': 'DoubleArray', 'BlueVTimeCourse': 'DoubleArray', 't1V': 'Double', 't2V': 'Double', 
#     't3V': 'Double', 'a1V': 'Double', 'a2V': 'Double', 'a3V': 'Double', 'n1V': 'Double', 'n2V': 'Double', 
#     'n3V': 'Double', 'tVOffset': 'Double', 'dotV': 'Double', 'dot2V': 'Double', 'srmV': 'Double', 'rlV': 'Double', 
#     'amp1V': 'Double', 'amp2V': 'Double', 'amp3V': 'Double', 'bluenessV': 'Double', 'Auto': 'DoubleArray', 
#     'acfBinning': 'Double', 'nSpikes': 'Double', 'acfMean': 'Double', 'acfRMS': 'Double', 'contamination': 'Double'}

# class ParamsWriter(object):
#     PARAM_FIELDS = {
#         'ID': 'Double', 'classID': 'String', 'x0': 'Double', 'y0': 'Double', 'SigmaX': 'Double', 'SigmaY': 'Double', 
#         'Theta': 'Double', 'gAmp': 'Double', 'contourX': 'DoubleArray', 'contourY': 'DoubleArray', 'contourArea': 'Double', 
#         'simpleContourX': 'DoubleArray', 'simpleContourY': 'DoubleArray', 'simpleContourArea': 'Double', 'EIx0': 'Double', 
#         'EIy0': 'Double', 'EISigmaX': 'Double', 'EISigmaY': 'Double', 'EITheta': 'Double', 'EIgAmp': 'Double', 
#         'RedTimeCourse': 'DoubleArray', 'GreenTimeCourse': 'DoubleArray', 'BlueTimeCourse': 'DoubleArray', 't1': 'Double', 
#         't2': 'Double', 't3': 'Double', 'a1': 'Double', 'a2': 'Double', 'a3': 'Double', 'n1': 'Double', 'n2': 'Double', 
#         'n3': 'Double', 'tOffset': 'Double', 'dot': 'Double', 'dot2': 'Double', 'srm': 'Double', 'rl': 'Double', 
#         'amp1': 'Double', 'amp2': 'Double', 'amp3': 'Double', 'blueness': 'Double', 'RedVTimeCourse': 'DoubleArray', 
#         'GreenVTimeCourse': 'DoubleArray', 'BlueVTimeCourse': 'DoubleArray', 't1V': 'Double', 't2V': 'Double', 
#         't3V': 'Double', 'a1V': 'Double', 'a2V': 'Double', 'a3V': 'Double', 'n1V': 'Double', 'n2V': 'Double', 
#         'n3V': 'Double', 'tVOffset': 'Double', 'dotV': 'Double', 'dot2V': 'Double', 'srmV': 'Double', 'rlV': 'Double', 
#         'amp1V': 'Double', 'amp2V': 'Double', 'amp3V': 'Double', 'bluenessV': 'Double', 'Auto': 'DoubleArray', 
#         'acfBinning': 'Double', 'nSpikes': 'Double', 'acfMean': 'Double', 'acfRMS': 'Double', 'contamination': 'Double'}
#     DOUBLE_SIZE = 10
#     STRING_SIZE = 9
    
#     def __init__(self, filepath, cluster_id):
#         """
#         Constructor method
#         Parameters:
#             filepath: Full path to the .params file that you intend to write
#             cluster_id: List of cluster ids that align with those contained in the .neurons file
#         """

#         self.filepath = filepath
#         self.cluster_id = cluster_id
#         # Parameter names and data types.
#         self.paramNames = self.getParamDict()
#         self.data_sizes = [10, 9, 10, 10, 10, 10, 10, 10, 6, 6, 10, 6, 6, 10, 408, 408, 408, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 6, 6, 6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1608, 10, 10, 10, 10]
#         self.fp = None

#     def write(self, timecourse_matrix, isi, spike_count, x0, y0, sigma_x, sigma_y, 
#               theta: np.ndarray=None, isi_binning: float=0.5, hull_vertices: np.ndarray=None, hull_area: np.ndarray=None):
#         """
#         Arguments:
#             timecourse_matrix: Timecourse matrix/temporal filters. Dimensions:[cell,time,RGB]  (type: numpy.ndarray)
#             isi: Interspike interval distributions. Dimensions:[cell, time] (type: numpy.ndarray)
#             spike_count: Total spike counts to noise stimulus. Dimensions:[cell] (type: numpy.ndarray)
#             x0: X-location (stixels) of RF center (type: numpy.ndarray)
#             y0: Y-location (stixels) of RF center (type: numpy.ndarray)
#             sigma_x: X-standard deviation of RF Gaussian (type: numpy.ndarray)
#             sigma_y: Y-standard deviation of RF Gaussian (type: numpy.ndarray)
#             theta: Rotational angle of Gaussian RF (in degrees) (type: numpy.ndarray)
#             isi_binning: Bin width (in milliseconds) used for computing the ISI distribution (type: float, default: 0.5)
#             hull_vertices: Vertices of the convex hull of the RF (type: numpy.ndarray)
#             hull_area: Area of the convex hull of the RF (type: numpy.ndarray)
#         """
#         num_params = len(self.paramNames)
#         num_cells = timecourse_matrix.shape[0]
#         maxNeurons = 10000
#         cellNameString = [0, 0, 0, 3, 65, 108, 108]

#         # Set the data sizes based on the inputs.
#         num_timecourse = timecourse_matrix.shape[1]
#         num_isi = isi.shape[1]
#         self.set_data_sizes(num_timecourse=num_timecourse, num_isi=num_isi)

#         # Open the file for writing.
#         with open(self.filepath, 'wb') as fp:
#             # Write the header
#             fp.write( struct.pack('>III', num_params, num_cells, maxNeurons) )

#             for param in self.paramNames:
#                 for i in range(3):
#                     fp.write(struct.pack('>b', 0))
#                 l = len(param)
#                 fp.write(struct.pack('>b', l))
#                 for k in range(l):
#                     fp.write(struct.pack('>s', bytes(param[k], 'utf-8')))
#                 for i in range(3):
#                     fp.write(struct.pack('>b', 0))
#                 value = self.paramNames[param]
#                 l = len(value)
#                 fp.write(struct.pack('>b', l))
#                 for k in range(l):
#                     fp.write(struct.pack('>s', bytes(value[k], 'utf-8')))

#             # Get the data locations.
#             locs = self.get_data_locations(num_params, num_cells)

#             # Write the locations.
#             fp.seek(num_params*21-1, os.SEEK_SET)
#             for loc in locs:
#                 fp.write(struct.pack('>I', loc-2))

#             end_of_fid = locs[-1]+8
#             num = end_of_fid - fp.tell()
#             for i in range(num):
#                 fp.write(struct.pack('>b', 0))

#             # Write the parameter values for each cell.
#             param_count = 0
#             for cellID in range(num_cells):
#                 for i in range(num_params):
#                     fp.seek(locs[param_count]-2, os.SEEK_SET)
#                     param_count += 1
#                     param_key = list(self.paramNames.keys())[i]
#                     param_value = list(self.paramNames.values())[i]
#                     if param_value == 'String':
#                         fp.write(struct.pack('>h', 327))
#                         for v in cellNameString:
#                             fp.write(struct.pack('>b', v))
#                     elif list(self.paramNames.values())[i] == 'DoubleArray':
#                         is_array = False
#                         if (param_key == 'RedTimeCourse' or 
#                             param_key == 'GreenTimeCourse' or
#                             param_key == 'BlueTimeCourse'):
#                             fp.write(struct.pack('>h', 319))
#                             fp.write(struct.pack('>h', 404))
#                             fp.write(struct.pack('>h', 0))
#                             fp.write(struct.pack('>h', num_timecourse))
#                             is_array = True
#                         elif param_key == 'Auto':
#                             fp.write(struct.pack('>h', 319))
#                             fp.write(struct.pack('>h', 1604))
#                             fp.write(struct.pack('>h', 0))
#                             fp.write(struct.pack('>h', num_isi))
#                             is_array = True
#                         else:
#                             fp.write(struct.pack('>h', 260))
#                             fp.write(struct.pack('>h', 0))
#                             fp.write(struct.pack('>h', 0))
                        
#                         if is_array:
#                             if param_key == 'RedTimeCourse':
#                                 my_array = timecourse_matrix[cellID, :, 0]
#                             elif param_key == 'GreenTimeCourse':
#                                 my_array = timecourse_matrix[cellID, :, 1]
#                             elif param_key == 'BlueTimeCourse':
#                                 my_array = timecourse_matrix[cellID, :, 2]
#                             else:
#                                 my_array = isi[cellID, :]
#                             for val in my_array:
#                                 fp.write(struct.pack('>d', val))
#                     else:
#                         if param_key == 'ID':
#                             k = self.cluster_id[cellID] #cellID + 1
#                         elif param_key == 'x0':
#                             k = np.max([1.0, x0[cellID]])
#                         elif param_key == 'y0':
#                             k = np.max([1.0, y0[cellID]])
#                         elif param_key == 'SigmaX':
#                             k = np.max([1.0, sigma_x[cellID]])
#                         elif param_key == 'SigmaY':
#                             k = np.max([1.0, sigma_y[cellID]])
#                         elif param_key == 'Theta':
#                             if theta is None:
#                                 k = 0.0
#                             else:
#                                 k = theta[cellID]
#                         elif param_key == 'nSpikes':
#                             k = spike_count[cellID]
#                         elif param_key == 'tOffset':
#                             k = 0.0
#                         elif param_key == 'acfBinning':
#                             k = isi_binning
#                         elif param_key == 'acfMean':
#                             k = np.mean(isi[cellID, :])
#                         elif param_key == 'acfRMS':
#                             k = 43.1999
#                         elif (param_key == 'blueness' or param_key == 'amp1' or param_key == 'amp2' or param_key == 'amp3'):
#                             k = 0.0
#                         else:
#                             k = 0.0
                        
#                         fp.write(struct.pack('>h', 200))
#                         fp.write(struct.pack('>d', k))
    
#     def getParamDict(self):
#         param_keys =  ['ID', 'classID', 'x0', 'y0', 'SigmaX', 'SigmaY', 'Theta', 'gAmp', 'contourX', 'contourY', 'contourArea', 'simpleContourX', 'simpleContourY', 'simpleContourArea', 'RedTimeCourse', 'GreenTimeCourse', 'BlueTimeCourse', 't1', 't2', 't3', 'a1', 'a2', 'a3', 'n1', 'n2', 'n3', 'tOffset', 'dot', 'dot2', 'srm', 'rl', 'amp1', 'amp2', 'amp3', 'blueness', 'RedVTimeCourse', 'GreenVTimeCourse', 'BlueVTimeCourse', 't1V', 't2V', 't3V', 'a1V', 'a2V', 'a3V', 'n1V', 'n2V', 'n3V', 'tVOffset', 'dotV', 'dot2V', 'srmV', 'rlV', 'amp1V', 'amp2V', 'amp3V', 'bluenessV', 'Auto', 'isiBinning', 'nSpikes', 'isiMean', 'isiRMS']
#         param_values = ['Double', 'String', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'DoubleArray', 'DoubleArray', 'Double', 'DoubleArray', 'DoubleArray', 'Double', 'DoubleArray', 'DoubleArray', 'DoubleArray', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'DoubleArray', 'DoubleArray', 'DoubleArray', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'Double', 'DoubleArray', 'Double', 'Double', 'Double', 'Double']

#         paramNames = dict()
#         for i, key in enumerate(param_keys):
#             paramNames[key] = param_values[i]
#         return paramNames

#     def get_data_locations(self, num_params, num_cells):
#         locs = []
#         locs.append(2441282)
#         count = 0
#         for cell in range(num_cells):
#             for param in range(num_params):
#                 locs.append(locs[count] + self.data_sizes[param])
#                 count += 1
#         locs = locs[:-1]
#         return locs

#     def set_data_sizes(self, num_timecourse: int=61, num_isi: int=300):
#         self.data_sizes = [10, 9, 10, 10, 10, 10, 10, 10, 6, 6, 10, 6, 6, 10, 408, 408, 408, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 6, 6, 6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1608, 10, 10, 10, 10]

#         # Set the timecourse length.
#         for i in range(14, 17):
#             self.data_sizes[i] = (num_timecourse+1)*8
        
#         self.data_sizes[56] = (num_isi+1)*8; # ACF

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         if self.fp is not None:
#             self.fp.close()

#     def close(self):
#         if self.fp is not None:
#             self.fp.close()




