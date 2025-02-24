import re
import os
import gzip
import json
import shutil
import imageio
import pydicom
import subprocess
from glob import glob
from pprint import pprint
from scipy import ndimage
from datetime import datetime
from itertools import groupby
from tqdm import tqdm
import bvbabel
import nibabel as nb

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence


class BVPP():

    def __init__(self, bv, project_folder, subject_id, target_folder=None):

        self.subject_id = subject_id
        if target_folder == None:
            target_folder = f'/{subject_id}_PP/'

        self.bv = bv

        self.project_folder = project_folder
        self.target_folder = target_folder
        self.data_folder = f'{project_folder}/RAW/'
        self.target_folder = self.project_folder + self.target_folder
        self.raw_folder = self.target_folder + '/fmr/'
        self.vmr_folder = self.target_folder + '/vmr/'
        self.scan_time_corrected_folder = self.target_folder + '/fmr_SCSTBL/'
        self.motion_corrected_folder = self.target_folder + '/fmr_SCSTBL_3DMCTS/'
        self.high_pass_folder = self.target_folder + '/fmr_SCSTBL_3DMCTS_HPF/'
        self.nii_folder = self.target_folder + '/nii/'
        self.topup_folder = self.target_folder + '/fmr_topup/'
        self.vtc_folder = self.target_folder + '/vtc/'
        self.use_nordic = False

        if not os.path.exists(self.target_folder):
            os.mkdir(self.target_folder)
        # change folder so that brainvoyager doesn't get confused.
        os.chdir(self.target_folder)



    def create_fmr(self,
                   project_dict,
                   data_folder=None,
                   target_folder=None,
                   skip_volumes_end=5,
                   skip_volumes_start=0,
                   first_5_vols=False,
                   use_nordic=False,
                   output_filename=None):

        if data_folder == None: data_folder = self.data_folder
        if target_folder == None: target_folder = self.target_folder

        # brainvoyager sometimes changes the working folder
        os.chdir(data_folder)

        # extract information from project dict
        dicom_filename = project_dict['filename']
        run = project_dict['run']
        condition = project_dict['condition']
        subject_id = project_dict['subject_id']
        signal = project_dict['signal']

        # nordic specific configuration
        if use_nordic:
            skip_volumes_end = 0
            nordic = '_nordic'
        else:
            nordic = ''

        # dicom information
        ds = pydicom.read_file(f'{data_folder}/{dicom_filename}')

        # set number of volumes
        prefix = dicom_filename.split('/')[-1].split('-')[0]

        n_volumes = self.get_n_volumes(data_folder, prefix, ds) - skip_volumes_end

        first_volume_amr = False

        n_slices = ds[0x19, 0x100a].value

        fmr_stc_filename = f'{subject_id}_{condition}_{run}_{signal}{nordic}'

        big_endian = not ds.is_little_endian

        mosaic_rows = ds.Rows

        mosaic_cols = ds.Columns

        slice_rows = ds.AcquisitionMatrix[0]

        slice_cols = ds.AcquisitionMatrix[-1]

        bytes_per_pixel = 2

        # check if it should be the AP file
        if first_5_vols:
            n_volumes = 5
            fmr_stc_filename += '_AP'

        # change the folder back because bv changes it for some reason.
        os.chdir(data_folder)

        if not os.path.exists(target_folder + '/fmr'):
            os.mkdir(target_folder + '/fmr')

        # create actual fmr files
        fmr_doc = self.bv.create_mosaic_fmr(data_folder + dicom_filename,
                                            n_volumes,
                                            skip_volumes_start,
                                            first_volume_amr,
                                            n_slices,
                                            fmr_stc_filename,
                                            big_endian,
                                            mosaic_rows,
                                            mosaic_cols,
                                            slice_rows,
                                            slice_cols,
                                            bytes_per_pixel,
                                            target_folder + '/fmr')

        try:
            filename = fmr_doc.file_name
            fmr_doc.close()
            return filename
        except Exception as e:
            print(e)
            return None

    def create_vmr(self,
                   data_dir,
                   vmr_dict,
                   target_folder):

        if not os.path.exists(target_folder + '/vmr'):
            os.mkdir(target_folder + '/vmr')
        os.chdir(target_folder + '/vmr')

        dicom_filename = vmr_dict['filename']
        subject_id = vmr_dict['subject_id']
        contrast_img = vmr_dict['contrast_img']

        ds = pydicom.read_file(f'{data_dir}/{dicom_filename}')

        scanner_file_type = "DICOM"
        n_slices = self.get_n_slices(data_dir, dicom_filename)

        first_file = f'{data_dir}/{dicom_filename}'
        big_endian = not ds.is_little_endian
        slice_rows = ds.Rows
        slice_cols = ds.Columns
        bytes_per_pixel = 2

        vmr_doc = self.bv.create_vmr(scanner_file_type,
                                     first_file,
                                     n_slices,
                                     big_endian,
                                     slice_rows,
                                     slice_cols,
                                     bytes_per_pixel)

        vmr_doc.save_as(f'{target_folder + "/vmr"}/{subject_id}_{contrast_img}.vmr')

        vmr_filename = vmr_doc.file_name

        vmr_doc.close()

        return vmr_filename

    def validate_volume_count(self, fmr_dict, condition_filename ):
        with open(condition_filename) as f:
            tr_dict = f.read()
        tr_dict = json.loads(tr_dict.replace('\'', '\"'))

        inconsistent = False
        n_checked = 0
        for run, n_vols in tr_dict.items():
            try:
                n_readed = int(fmr_dict["run"+run]["n_volumes"])
            except Exception as e:
                print(f'{e.__class__.__name__}, skipping!')
                continue
            # print(n_vols+5, n_readed)
            if n_vols + 5 is not n_readed:
                # n_vols: is the number of volumes that should be there
                # n_readed: is the number of volumes from the fmr
                print(f'Warning! \n\t\tFMR for run{run} found {n_readed} vols, [expects {n_vols} ({n_vols+5})].')
                inconsistent = True
            
            n_checked += 1

        if inconsistent is False:
            print('All read functional volumes are consistent with the reference list.')
        print(f'Total checked: {n_checked}.')


    def get_n_volumes(self, data_dir, prefix, ds):
        # get number of volumes
        series_number = str(ds.SeriesNumber).zfill(4)
        n_volumes = len(glob(f'{data_dir}/{prefix}-{series_number}*.dcm'))
        return n_volumes

    # IMPORTANT, this only works for anatomical
    def get_n_slices(self, data_dir, dicom_filename):
        # get number of slices
        series_number = dicom_filename.split('-')[1]
        
        prefixes = np.unique([dicom_filename.split('/')[-1].split('-')[0] for dicom_filename in glob(data_dir + '/*.dcm')])
        prefix = [prefix for prefix in prefixes if prefix in dicom_filename][0] + '-' + series_number

        n_slices = len(glob(f'{data_dir}/{prefix}*'))
        return n_slices

    def create_fmr_dict(self, data_folder=None, use_nordic=False, save=False):
        if data_folder == None:
            data_folder = self.data_folder

        project_dict = {}

        v1i1 = glob(data_folder + '/*-0001-00001.dcm')
        for filename in tqdm(v1i1):
            ds = pydicom.read_file(filename)

            if 'run' in ds.SeriesDescription.lower() and \
               'SBRef' not in ds.SeriesDescription:

                if not use_nordic and ds.ImageType[2] == 'P': continue

                run = ds.SeriesDescription.split('_')[-1].lower()

                prefix = filename.split('/')[-1].split('-')[0]

                n_volumes = self.get_n_volumes(data_folder, prefix, ds)

                # if 'WhatOff' in ds.SeriesDescription:
                if n_volumes < 180:
                    condition = 'WhatOff'
                else:
                    condition = 'WhatOn'

                # we use filename as key because its unique
                try:
                    project_dict[filename.split('/')[-1]] = {'run': run,
                                              'condition': condition,
                                              'signal': ds.ImageType[2],
                                              'subject_id': filename[filename.index('S0'):filename.index('S0')+3] , # filename.split('/')[-1].split('_')[-1][:3],
                                              'filename': filename.split('/')[-1],
                                              'n_volumes': n_volumes,
                                              'description': ds.SeriesDescription}
                except Exception as e:
                    project_dict[filename.split('/')[-1]] = {'run': run,
                                              'condition': condition,
                                              'signal': ds.ImageType[2],
                                              'subject_id': filename[filename.index('S1'):filename.index('S1')+3] , # filename.split('/')[-1].split('_')[-1][:3],
                                              'filename': filename.split('/')[-1],
                                              'n_volumes': n_volumes,
                                              'description': ds.SeriesDescription}

        if save:
            with open(f'{save}/fmr_info.json', 'w') as outfile: json.dump(project_dict, outfile)
        
        return project_dict

    def load_fmr_dict(path):
        return

    def create_vmr_dict(self, data_dir, subject_id=None, save=False):
        # prefix = glob(data_dir + '/*')[0].split('/')[-1].split('-')[0]
        prefixes = np.unique([filename.split('/')[-1].split('-')[0] for filename in glob(data_dir + '/*.dcm')])
        suffix = r'-\d{4}-0001-00001.dcm'

        project_dict = {}
        for prefix in prefixes:

            v1i1 = [f for f in os.listdir(data_dir) if re.search(prefix + suffix, f)]

            
            for filename in tqdm(v1i1):
                contrast_img = None
                if subject_id is None:
                    try:
                        subject_id = filename[filename.index('S0'):filename.index('S0')+3]
                    except Exception as e:
                        subject_id = filename[filename.index('S1'):filename.index('S1')+3]

                ds = pydicom.read_file(f'{data_dir}/{filename}')
                if 'INV' in ds.SeriesDescription and \
                   'PHS' not in ds.SeriesDescription:
                   contrast_img = ds.SeriesDescription.split('_')[-1]
                elif 'T1_Images' in ds.SeriesDescription:
                   contrast_img = 'T1'

                elif 'UNI_Images' in ds.SeriesDescription:
                   contrast_img = 'UNI'
                else:
                    contrast_img = None

                if contrast_img:
                    project_dict[filename] = {'contrast_img': contrast_img,
                                              'subject_id': subject_id,
                                              'filename': filename,
                                              'description': ds.SeriesDescription,
                                              'n_volumes': self.get_n_volumes(data_dir, prefix, ds)}


        if save:
            with open(f'{save}/vmr_info.json', 'w') as outfile: json.dump(project_dict, outfile)
        
        return project_dict

    def parse_log_file(self, fmr_filename, keys):
        log_filename = fmr_filename.replace('3DMCTS.fmr', '3DMC.log')
        if len(log_filename) == 0:
            return 'empty'
        with open(log_filename) as f:
            log_file = f.readlines()[4:]  # ignore the log info for now

        parameter_set = []
        for line in log_file:
            vol_pars = line.strip('-> \n')

            vol_pars = list(
                filter(
                    lambda x: x not in ['', 'mm', 'degs'], vol_pars.split(' ')
                )
            )

            vol_pars = {k[:-1]: v for k, v in zip(vol_pars[::2],
                                                  vol_pars[1::2])}

            parameter_set.append(vol_pars)

        data = {}
        for key in keys:
            data[key] = [float(item[key]) for item in parameter_set]

        return data

    def apply_motion_correction_within_run(self, target_folder, fmr_stctbl_filenames):
        trilinear_with_motion_detection = 2

        processed_folder = target_folder + 'fmr_SCSTBL_3DMCTS_within/'
        if not os.path.exists(target_folder + 'fmr_SCSTBL_3DMCTS_within'):
            os.mkdir(target_folder + 'fmr_SCSTBL_3DMCTS_within')
                    
        fmr_mc_filenames = []
        for filename in tqdm(fmr_stctbl_filenames):
            fmr_doc = self.bv.open(filename, True)
            
            fmr_doc.correct_motion_to_run_ext(filename,
                                              0, # first volume
                                              trilinear_with_motion_detection,
                                              True, # full dataset
                                              200, # max iterations
                                              False, # create movie
                                              True) # extended log file
            
            try:
                preprocessed_fmr_name = shutil.move(fmr_doc.preprocessed_fmr_name, processed_folder)
                shutil.move(fmr_doc.preprocessed_fmr_name.replace('.fmr', '.stc'), processed_folder)
                shutil.move(fmr_doc.preprocessed_fmr_name.replace('3DMCTS.fmr', '3DMC.log'), processed_folder) # validate this!
                shutil.move(fmr_doc.preprocessed_fmr_name.replace('3DMCTS.fmr', '3DMC.sdm'), processed_folder) # validate this!

                fmr_mc_filenames.append(processed_folder + fmr_doc.preprocessed_fmr_name.split('/')[-1])
            except Exception as e:
                print(e)
            
            fmr_doc.close()

    def extract_volumes(self, fmr_mc_filenames, n_volumes=5):
        # extract first five volumes
        for filename in tqdm(fmr_mc_filenames):
            head, data = bvbabel.fmr.read_fmr(filename)

            suffix = f'_first{n_volumes}Volumes'
            out_filename = filename.replace('.fmr', suffix)
            data = data[..., :n_volumes]

            # write stc
            bvbabel.fmr.write_stc(out_filename+'.stc', data, data_type=head["DataType"])
            
            # write fmr
            with open(filename, 'r') as f:
                fmr_data = f.read()
                res = re.sub(r'Prefix:                        .*', f'Prefix:                        \"{filename.split("/")[-1][:-4]}{suffix}\"', fmr_data)
                res = re.sub(r'NrOfVolumes:                   .*', f'NrOfVolumes:                   {n_volumes}', res)
            
            with open(out_filename + '.fmr', 'w') as f:
                f.write(res)

    def cut_volumes(self, fmr_filenames, start=6, end=10, suffix='_cut'):
        for filename in fmr_filenames:
            print(f'processing {filename}')
            head, data = bvbabel.fmr.read_fmr(filename)

            out_filename = filename.replace('.fmr', suffix)
            data = data[:,:,start:-end,:]

            # write stc
            bvbabel.fmr.write_stc(out_filename+'.stc', data, data_type=head["DataType"])
            
            # write fmr
            with open(filename, 'r') as f:
                fmr_data = f.read()
                res = re.sub(r'Prefix:                        .*', f'Prefix:                        \"{filename.split("/")[-1][:-4]}{suffix}\"', fmr_data)
                res = re.sub(r'NrOfSlices:                    .*', f'NrOfSlices:                    {data.shape[2]}', res)
            
            with open(out_filename + '.fmr', 'w') as f:
                f.write(res)


    def calculate_volume_means(self, filenames):
        # calculate the means
        for filename in tqdm(filenames):
            head, data = bvbabel.fmr.read_fmr(filename)
            #nii_data = nb.load(filename).get_fdata()

            suffix = '_mean'
            out_filename = filename.replace('.fmr', suffix)
            data = np.expand_dims(np.mean(data, axis=-1), -1)

            # write stc
            bvbabel.fmr.write_stc(out_filename+'.stc', data, data_type=head["DataType"])
            
            # write fmr
            with open(filename, 'r') as f:
                fmr_data = f.read()
                res = re.sub(r'Prefix:                        .*', f'Prefix:                        \"{filename.split("/")[-1][:-4]}{suffix}\"', fmr_data)
                res = re.sub(r'NrOfVolumes:                   .*', f'NrOfVolumes:                   1', res)
            
            with open(out_filename+'.fmr', 'w') as f:
                f.write(res)
            # print(f'written file: {out_filename}')

    def apply_slice_scan_time_correction(self, fmr_filenames, target_folder, interpolation_method):
        os.chdir(target_folder + 'fmr/')

        processed_directory = target_folder + '/fmr_SCSTBL/'
        if not os.path.exists(target_folder + '/fmr_SCSTBL'):
            os.mkdir(target_folder + '/fmr_SCSTBL')
                    
        fmr_stctbl_filenames = []
        for filename in tqdm(fmr_filenames): # + [ap_filename, pa_filename]:
            fmr_doc = self.bv.open(filename)
            
            if not os.path.isfile(processed_directory + fmr_doc.file_name.split('/')[-1].replace('.fmr', '_SCSTBL.fmr')):
                fmr_doc.correct_slicetiming_using_timingtable(interpolation_method)
                preprocessed_fmr_name = shutil.move(fmr_doc.preprocessed_fmr_name, processed_directory)
                shutil.move(fmr_doc.preprocessed_fmr_name.replace('.fmr', '.stc'), processed_directory)
                fmr_stctbl_filenames.append(preprocessed_fmr_name)
            else:
                fmr_stctbl_filenames.append(processed_directory + fmr_doc.file_name.split('/')[-1].replace('.fmr', '_SCSTBL.fmr'))

            fmr_doc.close() 
        return fmr_stctbl_filenames       

    def load_mean_correction_values(self, file_list):
        mean_correction = {}
        for filename in file_list:
            try:
                with open(filename, 'r') as f:
                    transformations = f.readlines()[-1]
                    transformations = [float(x) for x in re.findall(r'[\d.-]+', transformations)]
                    mean_correction[filename.split('/')[-1]] = transformations
            except Exception as e:
                print(e)
                raise ValueError(f'caused by: {filename}')
        return mean_correction

    def apply_high_pass_filter(self, filenames, n_cycles):
        fmr_hpf_filenames = []
        for filename in tqdm(filenames):
            fmr_doc = self.bv.open(filename, True)
            fmr_doc.filter_temporal_highpass_glm_fourier(n_cycles)
            fmr_hpf_filenames.append(fmr_doc.preprocessed_fmr_name)
        return fmr_hpf_filenames


    def apply_motion_parameters(self, file_list, motion_correction_parameters):
        pbar = tqdm(file_list)
        for filename in pbar:
            pbar.set_description(filename.split('/')[-1])
            sdm_filename = filename.split('/')[-1].replace('.fmr', '_3DMCTS_first5Volumes_mean_3DMC.sdm')
            
            transformations = motion_correction_parameters[sdm_filename]
            # print('\t', 'with transformations:', str(transformations), end='\t')
            
            fmr_doc = self.bv.open(filename)
            within_sdm = self.target_folder + 'fmr_SCSTBL_3DMCTS_within/' + filename.split('/')[-1].replace('.fmr', '_3DMC.sdm')
            # print('\n\t and ', within_sdm)
            result = fmr_doc.apply_motion_params(within_sdm, 3, *transformations)
            # print(result)
            fmr_doc.close()

    def apply_motion_correction_between_means(self, filenames, first_volume):
        # apply motion correcton to the mean of the first five volumes
        mean_correction = {}
        for filename in tqdm(filenames):
            # print(filename)
            fmr_doc = self.bv.open_document(filename)
            fmr_doc.correct_motion_to_run_ext(first_volume, 0, 3, True, 200, False, True)
            fmr_doc.close()
            
            # print(filename.replace('_3DMCTS_first5Volumes_mean.fmr', '_3DMC.sdm'))
            with open(filename.replace('_3DMCTS_first5Volumes_mean.fmr', '_3DMC.sdm'), 'r') as f:
                transformations = f.readlines()[-1]
                transformations = [float(x) for x in re.findall(r'[\d.-]+', transformations)]
                mean_correction[filename.split('/')[-1]] = transformations
        # return mean_correction


    def apply_motion_correction_with_params(self, target_folder, fmr_stctbl_filenames, mean_correction):
        for filename in fmr_stctbl_filenames:
            #[sdm_filename for sdm_filename in list(mean_correction.keys()) if filename.split('_')[-2] in sdm_filename and filename.split('_')[-3] in sdm_filename][0]
            print(f'processing {filename}')

            transformations = mean_correction[filename.replace('.fmr', '_3DMCTS_first5Volumes_mean.fmr').split('/')[-1]]
            print('\t', 'with transformations:', str(transformations), end='\t')
            fmr_doc = self.bv.open(filename)
            within_sdm = target_folder + 'fmr_SCSTBL_3DMCTS_within/' + filename.split('/')[-1].replace('.fmr', '_3DMC.sdm')
            print('\n', within_sdm)
            result = fmr_doc.apply_motion_params(within_sdm, 3, *transformations)
            print(result)
            fmr_doc.close()


    def plot_motion_params(self, data, keys, title, ax=None):
        if not ax:
            fig, ax = plt.subplots()

        for key in keys:
            ax.plot(data[key], label=keys)
            ax.set_title(title)

    def plot_motion_params_all_runs(self, filenames, keys):
        fig, ax = plt.subplots(1,
                               len(filenames),
                               figsize=(2.5 * len(filenames), 2.5),
                               sharey=True)

        for idx, filename in enumerate(filenames):
            data = self.parse_log_file(filename, keys)
            if data == 'empty': 
                print('found empty data')
                continue
            try:
                title = filename.split('/')[-1].split('_')[2]
            except:
                title = filename.split('_')[2]

            self.plot_motion_params(data,
                                    keys,
                                    title,
                                    ax=ax[idx])

        ax[-1].legend(keys,
                      fontsize=10,
                      fancybox=True,
                      shadow=True,
                      bbox_to_anchor=(1, 1))
        return fig


    def plot_pre_post_motion_correction(self,
                                        slice,
                                        volume,
                                        post_mc_filenames,
                                        pre_mc_filenames,
                                        run1_first_volume):
        # plot before and after motion correction
        fig, ax = plt.subplots(2, len(post_mc_filenames), figsize=(2.5*len(post_mc_filenames), 5))

        for idx, (pre, post) in enumerate(zip(pre_mc_filenames, post_mc_filenames)):
            try:
                fmr_pre_mc_data = bvbabel.fmr.read_fmr(pre)[1]
            except Exception as e:
                print(e)
                print(pre)
                continue
            if not os.path.exists(post): continue
            fmr_post_mc_data = bvbabel.fmr.read_fmr(post)[1]
            try:    
                ax[0][idx].imshow(np.rot90(fmr_pre_mc_data[:,:,slice,volume] - run1_first_volume), cmap='gray')
                ax[0][idx].set_title(f'Run {idx+1}\n\npre MC')
                ax[0][idx].axis('off')

                ax[1][idx].imshow(np.rot90(fmr_post_mc_data[:,:,slice,volume] - run1_first_volume), cmap='gray')
                ax[1][idx].set_title('post MC')
                ax[1][idx].axis('off')
            except Exception as e:
                print(pre, e)

        return fig

    def create_gifs(self, data_dir, reference_run, other_run):
        os.chdir(data_dir)
        fig, ax = plt.subplots(9, 7, figsize=(3*7, 3*9))
        return fig

    def plot_distortion_correction_gifs(self,
                                        data_dir,
                                        pre_filenames,
                                        post_filenames,
                                        condition,
                                        slice,
                                        volume):
        os.chdir(data_dir)

        # pre
        fig, ax = plt.subplots(1, len(pre_filenames), figsize=(2.5*len(pre_filenames), 2.5))
        if len(pre_filenames) == 1:
            ax.axis('off')
            for idx, pre in enumerate(pre_filenames):
                fmr_pre_topup_data = bvbabel.fmr.read_fmr(pre)[1]
                ax.imshow(np.rot90(fmr_pre_topup_data[:,:,slice,volume]))
                ax.set_title(pre[pre.index('run'):pre.index('run')+4])
        else:
            [axis.axis('off') for axis in ax]
            for idx, pre in enumerate(pre_filenames):
                fmr_pre_topup_data = bvbabel.fmr.read_fmr(pre)[1]
                try:
                    ax[idx].imshow(np.rot90(fmr_pre_topup_data[:,:,slice,volume]))
                except:
                    continue
                ax[idx].set_title(pre[pre.index('run'):pre.index('run')+4])

        pre_image_filename = pre.split('/')[-1].split('.')[0] + '.png'
        plt.suptitle(f'Samples before and after distortion correction ({condition})')
        plt.tight_layout()
        plt.savefig(f'{condition}_pre_topup.png')

        # post
        fig, ax = plt.subplots(1, len(post_filenames), figsize=(2.5*len(post_filenames), 2.5))
        if len(pre_filenames) == 1:
            ax.axis('off')
            for idx, post in enumerate(post_filenames):
                fmr_post_topup_data = bvbabel.fmr.read_fmr(post)[1]
                ax.imshow(np.rot90(fmr_post_topup_data[:,:,slice,volume]))
                ax.set_title(post[post.index('run'):post.index('run')+4])
        else:
            [axis.axis('off') for axis in ax]
            for idx, post in enumerate(post_filenames):
                fmr_post_topup_data = bvbabel.fmr.read_fmr(post)[1]
                try:
                    ax[idx].imshow(np.rot90(fmr_post_topup_data[:,:,slice,volume]))
                except:
                    continue
                ax[idx].set_title(post[post.index('run'):post.index('run')+4])
        post_image_filename = post.split('/')[-1].split('.')[0] + '.png'
        plt.suptitle(f'Samples before and after distortion correction ({condition})')
        plt.tight_layout()
        plt.savefig(f'{condition}_post_topup.png')

        # save both gif and webp
        with imageio.get_writer(f'{condition}_topup.gif', mode='I') as writer:
            writer.append_data(imageio.imread(f'{condition}_pre_topup.png'))
            writer.append_data(imageio.imread(f'{condition}_post_topup.png'))

        sequence = []
        im = Image.open(f'{condition}_topup.gif')
        for frame in ImageSequence.Iterator(im): sequence.append(frame.copy())
        output_filename = f'{condition}_topup.webp'
        sequence[0].save(output_filename, save_all=True,  append_images = sequence[1:])

        return output_filename

    # fsl helper functions

    def convert_dcm_to_nifti(self, project_dir, data_dir, target_dir):
        if not os.path.exists(target_dir): os.mkdir(target_dir)
        cmd = f'{project_dir}/../../../Scripts/dcm2niix -o {target_dir} {data_dir}'
        print(subprocess.check_output(cmd.split(' ')))

    def convert_fmr_to_nifti(self, target_folder, filename):
        header, data = bvbabel.fmr.read_fmr(filename)
        basename = filename.split(os.extsep, 1)[0].split('/')[-1]
        outname = target_folder + f"{basename}_bvbabel.nii.gz"
        img = nb.Nifti1Image(data, affine=np.eye(4))
        nb.save(img, outname)
        print('saved to: ', outname)
        return outname

    def extract_first_volume(self, filename):
        data = nb.load(filename).get_fdata()
        data_1vol = data[..., 1]
        data_1vol = np.expand_dims(data_1vol, -1)

        new_filename = filename.replace('.nii', '_1vol.nii')
        img = nb.Nifti1Image(data_1vol, np.eye(4))
        nb.save(img, new_filename)
        print(f'File saved as {new_filename}')

    def convert_nifti_to_fmr(self, nifti_filename, fmr_filename, skip_volumes_end=5):
        # load nifti data
        nifti_data = nb.load(f'{nifti_filename}').get_fdata()
        # load old fmr header
        fmr_header = bvbabel.fmr.read_fmr(fmr_filename)[0]
        n_vols = fmr_header['NrOfVolumes']

        # copy nifti data to new stc
        new_stc_filename = fmr_filename.replace(".fmr", "_nonoise.stc")
        bvbabel.stc.write_stc(new_stc_filename, nifti_data, data_type=fmr_header["DataType"])

        # change the prefix and write fmr in new file
        new_fmr_filename = fmr_filename.replace('.fmr', '_nonoise.fmr')

        fmr_data = open(fmr_filename, 'r').read()
        fmr_filename = fmr_filename.split('/')[-1]
        fmr_data = fmr_data.replace(fmr_filename.split('.')[0],
                                    new_fmr_filename.split('.')[0].split('/')[-1])
        fmr_data = fmr_data.replace(str(n_vols), str(n_vols - skip_volumes_end), 1)

        new_fmr_file = open(new_fmr_filename, 'w')
        new_fmr_file.write(fmr_data)
        new_fmr_file.close()
        return new_fmr_filename

    def convert_nifti_to_fmr2(self, target_folder, nifti_filename, fmr_filename):
        nifti_data = nb.load(nifti_filename).get_fdata()
        fmr_header = bvbabel.fmr.read_fmr(fmr_filename)[0]
        nifti_filename = nifti_filename.split('/')[-1]
        new_stc_filename = f'{target_folder}/{nifti_filename.replace(".nii.gz", ".stc")}'
        bvbabel.stc.write_stc(new_stc_filename, nifti_data, data_type=fmr_header["DataType"])

        # change the prefix and write fmr in new file
        new_fmr_filename = f'{target_folder}/{nifti_filename.replace(".nii.gz", ".fmr")}'
        fmr_data = open(fmr_filename, 'r').read()
        fmr_filename = fmr_filename.split('/')[-1]
        fmr_data = fmr_data.replace(fmr_filename.split('.')[0], nifti_filename.split('.')[0])
        new_fmr_file = open(new_fmr_filename, 'w')
        new_fmr_file.write(fmr_data)
        new_fmr_file.close()
        return new_fmr_filename

    def fsl_merge(self, target_folder, file1, file2, subject_id, run_idx, pattern='APPA'):
        env = dict(os.environ)
        env['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

        output_filename = f'{subject_id}_{run_idx}_{pattern}.nii.gz'

        if pattern == 'APPA':
            cmd = f'/usr/local/fsl/bin/fslmerge -t {target_folder}{output_filename} {file1} {file2}'
        elif pattern == 'PAAP':
            cmd = f'/usr/local/fsl/bin/fslmerge -t {target_folder}{output_filename} {file2} {file1}'
        else:
            raise ValueError(f'parameter "{pattern}" is not accepted!')

        try:
            output = subprocess.check_output(cmd.split(' '),
                                             stderr=subprocess.STDOUT,
                                             env=env)
            print(output)
        except Exception as e:
            print(e)
        return output_filename

    def fsl_topup(self, filename, acqiusion_parameters_filename, config_filename):
        output_filename = filename.split(os.extsep, 1)[0] + '_topup'

        env = dict(os.environ)
        env['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

        cmd = f'/usr/local/fsl/bin/topup --imain={filename} ' \
              f'--datain={acqiusion_parameters_filename} ' \
              f'--config={config_filename} ' \
              f'--out={output_filename}'
        # print(cmd)
        try:
            output = subprocess.check_output(cmd.split(' '), 
                                             stderr=subprocess.STDOUT,
                                             env=env)
            print(output)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        return f'{output_filename}_fieldcoef.nii.gz'

    def fsl_apply_topup(self,
                        data_dir,
                        filename,
                        acqiusion_parameters_filename,
                        topup_filename
                        ):

        env = dict(os.environ)
        env['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

        output_filename = filename.split(os.extsep, 1)[0] + '_topup'

        cmd = '/usr/local/fsl/bin/applytopup ' \
              f'-i {filename} ' \
              f'-a {acqiusion_parameters_filename} ' \
              f'-t {topup_filename} ' \
              f'-x 1 -m jac -v ' \
              f'-o {output_filename}'
        os.chdir(data_dir)

        try:
            output = subprocess.check_output(cmd.split(' '), stderr=subprocess.STDOUT, env=env)
            pprint(output)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

        return f'{output_filename}.nii.gz'

    def create_acquisition_parameters(self, target_folder, pattern='APPA', n_volumes=5):
        # this is really hardcoded for now!
        filename = 'acqparams.txt'

        ap_phase_encoding = '0 1 0 1\n'
        pa_phase_encoding = '0 -1 0 1\n'

        if pattern == 'APPA':
            acqiusion_parameters = ap_phase_encoding * n_volumes + \
                                   pa_phase_encoding * n_volumes
        elif pattern == 'PAAP':
            acqiusion_parameters = pa_phase_encoding * n_volumes + \
                                   ap_phase_encoding * n_volumes
        else:
            raise ValueError(f'parameter "{pattern}" is not accepted!')

        with open(f'{target_folder}/{filename}', 'w') as f:
            f.write(acqiusion_parameters)

        return filename

    def create_b02b0_cnf(self, target_folder):
        filename = 'b02b0.cnf'
        b0cnf = '''# Resolution (knot-spacing) of warps in mm\n
                   --warpres=20,16,14,12,10,6,4,4,4\n
                   # Subsampling level (a value of 2 indicates that a 2x2x2 neighbourhood is collapsed to 1 voxel)\n
                   --subsamp=2,2,2,2,2,1,1,1,1\n
                   # FWHM of gaussian smoothing\n
                   --fwhm=8,6,4,3,3,2,1,0,0\n# Maximum number of iterations\n
                   --miter=5,5,5,5,5,10,10,20,50\n# Relative weight of regularisation\n
                   --lambda=0.005,0.001,0.0001,0.000015,0.000005,0.0000005,0.00000005,0.0000000005,0.00000000001\n
                   # If set to 1 lambda is multiplied by the current average squared difference\n
                   --ssqlambda=1\n
                   # Regularisation model\n
                   --regmod=bending_energy\n
                   # If set to 1 movements are estimated along with the field\n
                   --estmov=0,0,0,0,0,0,0,0,0\n
                   # 0=Levenberg-Marquardt, 1=Scaled Conjugate Gradient\n
                   --minmet=0,0,0,0,0,1,1,1,1\n
                   # Quadratic or cubic splines\n
                   --splineorder=3\n
                   # Precision for calculation and storage of Hessian\n
                   --numprec=double\n
                   # Linear or spline interpolation\n
                   --interp=spline\n
                   # If set to 1 the images are individually scaled to a common mean intensity \n
                   --scale=1'''

        with open(f'{target_folder}/{filename}', 'w') as f:
            f.write(b0cnf)  

        return filename      

    # anatomical preprocessing and coregistration

    def mp2rage_genuniden(self,
                          data_dir,
                          chosen_factor,
                          filenameUNI='UNI.v16',
                          filenameINV1='INV1.v16',
                          filenameINV2='INV2.v16',
                          uniden_output_filename='uniden.v16',
                          savev16=True,
                          savevmr=True):
        """function to take mp2rage files uni, inv1 and inv2 and given a chosen denoising factor
        returns denoised images"""
        
        ############################################
        # Note: Python implemention of matlab code https://github.com/khanlab/mp2rage_genUniDen.git mp2rage_genUniDen.m
        # Date: 2019/09/25
        # Author: original author YingLi Lu, adopted by Jorie van Haren - translated for brainvoyager
        ############################################

        filepath = lambda filename: '{}/{}'.format(data_dir, filename)

        # load data
        header, mp2rage_img = bvbabel.v16.read_v16(filepath(filenameUNI))
        _, inv1_img = bvbabel.v16.read_v16(filepath(filenameINV1))
        _, inv2_img = bvbabel.v16.read_v16(filepath(filenameINV2))

        # adjust dimensions for slight mismatches of phase data
        if inv1_img.shape != mp2rage_img.shape:
            tp_img = np.zeros(mp2rage_img.shape)
            tp_img[:inv1_img.shape[0], :inv1_img.shape[1], :inv1_img.shape[2]] = inv1_img
            inv1_img = tp_img
        if inv2_img.shape != mp2rage_img.shape:
            tp_img = np.zeros(mp2rage_img.shape)
            tp_img[:inv2_img.shape[0], :inv2_img.shape[1], :inv2_img.shape[2]] = inv2_img
            inv2_img = tp_img

        mp2rage_img = mp2rage_img.astype('float64')
        inv1_img = inv1_img.astype('float64')
        inv2_img = inv2_img.astype('float64')

        if mp2rage_img.min() >= 0 and mp2rage_img.max() >= 0.51:
           # converts MP2RAGE to -0.5 to 0.5 scale - assumes that it is getting only positive values
            mp2rage_img = (
                mp2rage_img - mp2rage_img.max()/2)/mp2rage_img.max()
            integerformat = 1
        else:
            integerformat = 0

        # computes correct INV1 dataset
        inv1_img = np.sign(mp2rage_img)*inv1_img # gives the correct polarity to INV1

        # because the MP2RAGE INV1 and INV2 is a sum of squares data, while the
        # MP2RAGEimg is a phase sensitive coil combination.. some more maths has to
        # be performed to get a better INV1 estimate which here is done by assuming
        # both INV2 is closer to a real phase sensitive combination

        # INV1pos=rootsquares_pos(-MP2RAGEimg.img,INV2img.img,-INV2img.img.^2.*MP2RAGEimg.img);
        inv1pos = self.rootsquares_pos(-mp2rage_img, inv2_img, -inv2_img**2*mp2rage_img)
        inv1neg = self.rootsquares_neg(-mp2rage_img, inv2_img, -inv2_img**2*mp2rage_img)

        inv1final = inv1_img

        inv1final[np.absolute(inv1_img-inv1pos) > np.absolute(inv1_img-inv1neg)] = inv1neg[np.absolute(inv1_img-inv1pos) > np.absolute(inv1_img-inv1neg)]
        inv1final[np.absolute(inv1_img-inv1pos) <= np.absolute(inv1_img-inv1neg)] = inv1pos[np.absolute(inv1_img-inv1pos) <= np.absolute(inv1_img-inv1neg)]

        # usually the multiplicative factor shouldn't be greater then 10, but that
        # is not the case when the image is bias field corrected, in which case the
        # noise estimated at the edge of the imagemight not be such a good measure

        multiplyingFactor = chosen_factor
        noiselevel = multiplyingFactor*np.mean(inv2_img[:, -11:, -11:])

        # run the actual denoising function
        mp2rage_imgRobustPhaseSensitive = self.mp2rage_robustfunc( inv1final, inv2_img, noiselevel**2)

        # set to interger format
        mp2rageimg_img = np.round(4095*(mp2rage_imgRobustPhaseSensitive+0.5))

        # Save image
        mp2rageimg_img = nb.casting.float_to_int(mp2rageimg_img,'int16');
        if savev16: bvbabel.v16.write_v16(filepath(uniden_output_filename), header, mp2rageimg_img)
        if savevmr: 
            mp2rageimg_img = np.uint8(np.round(225*(mp2rage_imgRobustPhaseSensitive+0.5)))
            vmrheader, _ = bvbabel.vmr.read_vmr(filepath('{}.vmr'.format(re.search(r'.+(?=\.)',filenameUNI)[0])))
            bvbabel.vmr.write_vmr(filepath('{}.vmr'.format(re.search(r'.+(?=\.)',uniden_output_filename)[0])), vmrheader, mp2rageimg_img)

        return(header, mp2rageimg_img)

    def mp2rage_robustfunc(self, INV1, INV2, beta):
        """adaptation of matlab robust denoise function"""
        return (np.conj(INV1)*INV2-beta)/(INV1**2+INV2**2+2*beta)

    def rootsquares_pos(self, a, b, c):
        # matlab:rootsquares_pos=@(a, b, c)(-b+sqrt(b. ^ 2 - 4 * a.*c))./(2*a)
        return (-b+np.sqrt(b**2 - 4*a*c))/(2*a)


    def rootsquares_neg(self, a, b, c):
        # matlab: rootsquares_neg = @(a, b, c)(-b-sqrt(b. ^ 2 - 4 * a.*c))./(2*a)
        return (-b-np.sqrt(b**2 - 4*a*c))/(2*a)


    def apply_intensity_mask(self,
                             data_dir, mask_fn='segmentation_mask',
                             uniden_fn='uniden_IIHC', 
                             cortex_in=1,     # set cortex mask intensity
                             cerebellum_in=1, # set cerebellum mask intensity
                             csf_in=1,        # set cfs mask intensity
                             skull_in=0.5,   # set skull mask intensity
                             background_in=0):# set background noise mask intesity

        # set filepath lambda for data_folder
        filepath = lambda filename: '{}/{}'.format(data_dir, filename)

        # load masking segmentation
        _, result = bvbabel.vmr.read_vmr(filepath('{}.vmr'.format(mask_fn)))

        # load the files to apply the soft mask to
        header, img = bvbabel.vmr.read_vmr(filepath('{}.vmr'.format(uniden_fn)))
        headerv16, imgv16 = bvbabel.vmr.read_vmr(filepath('{}.v16'.format(uniden_fn)))

        # create empty mask array
        mask = np.zeros(img.shape)

        mask[result == 1] = cortex_in
        mask[result == 2] = skull_in
        mask[result == 3] = cerebellum_in
        mask[result == 4] = csf_in
        mask[result == 5] = background_in
        
        # then for img, and v16 img apply intensity mask
        img = img * mask
        img = img.astype(np.uint8)
        imgv16 = imgv16 * mask
        imgv16 = imgv16.astype(np.uint16)

        # save the files in bv format
        bvbabel.vmr.write_vmr(filepath('{}_masked.vmr'.format(uniden_fn)), header, img)
        bvbabel.vmr.write_vmr(filepath('{}_masked.v16'.format(uniden_fn)), headerv16, imgv16)
        return(bv.open_document(filepath('{}_masked.vmr'.format(uniden_fn))))
        

    def correct_inhomogeneities(self,
                                data_dir,
                                doc_vmr_fn='uniden.vmr', 
                                actract_brain = True,       # bool whether to include skull stripping step
                                n_cycles = 8,              # number of itterations for fitting bias field
                                wm_tissue_range = 0.25,     # threshold to detect whether regions contain one or two tissue types
                                wm_intensity_thresh = 0.3,   # threshold to seperate wm from gm
                                polynom_order = 3):         # order of polynom to fit 3d field
        """correct inhomonegeity using brainvoyagers extended function, 
        then open up adjusted volume"""

        doc_vmr = self.bv.open_document(f'{data_dir}/{doc_vmr_fn}')

        # get new filename
        fn = '{}_IIHC.vmr'.format(re.search(r'.+(?=\.)',doc_vmr.file_name)[0])

        # do inhomonegeity correction
        print('\nRunning inhomonegeity correction for: {}'.format(doc_vmr.file_name))
        doc_vmr.correct_intensity_inhomogeneities_ext(actract_brain, n_cycles, wm_tissue_range, wm_intensity_thresh, polynom_order)
        doc_vmr = self.bv.open_document(f'{data_dir}/{doc_vmr_fn.split(".")[0]}_IIHC.vmr')
        
        return(doc_vmr)

    def erode_mask(self, data_dir, uniden_filename='uniden', iterations=6):
        """erode the mask created in the homogenity correction step
        the idea is to get rid of the peel (and probably some of the grey matter) but leaving wm in tact"""

        # load mask
        maskhead, maskimg = bvbabel.vmr.read_vmr(f'{data_dir}/{uniden_filename}')

        # set mask to binary
        mask = np.zeros(maskimg.shape)
        mask[maskimg > 0] = 1

        # erode the mask in bool and put back in original masking array
        mask = ndimage.binary_erosion(mask, iterations=iterations).astype(np.uint8)
        maskimg[mask == 0] = 0
        bvbabel.vmr.write_vmr(f'{data_dir}/{uniden_filename}', maskhead, maskimg)
        

    def apply_erosion_mask(self,
                           data_dir,
                           mask_fn='uniden_BrainMask',
                           uniden_fn='uniden_IIHC',
                           outmask_int=0):
        """the intensity to apply to all the things outside the new mask"""
        
        # load eroded masking
        _, mask = bvbabel.vmr.read_vmr(f'{data_dir}/{mask_fn}')

        # load the files to apply the soft mask to
        header, img = bvbabel.vmr.read_vmr(f'{data_dir}/{uniden_fn}')
        headerv16, imgv16 = bvbabel.v16.read_v16(f'{data_dir}/{uniden_fn.split(".")[0]}.v16')

        # set mask values
        mask[mask == 0] = outmask_int
        mask[mask == np.max(mask)] = 1
        
        # then for img, and v16 img apply intensity mask
        img = img * mask
        img = img.astype(np.uint8)
        imgv16 = imgv16 * mask
        imgv16 = imgv16.astype(np.uint16)

        # save the files in bv format
        uniden_fn = uniden_fn.split('.')[0]
        bvbabel.vmr.write_vmr(f'{data_dir}/{uniden_fn.split(".")[0]}_masked.vmr', header, img)
        bvbabel.v16.write_v16(f'{data_dir}/{uniden_fn.split(".")[0]}_masked.v16', headerv16, imgv16)
        return(self.bv.open_document(f'{data_dir}/{uniden_fn.split(".")[0]}_masked.vmr'))
    

    def isovoxel(self,
                 data_dir,
                 doc_vmr_fn, 
                 res = 0.4,       # target resolution
                 framing_cube = 512,  # framing dimensions of output vmr data
                 interpolation = 2,   # interpolation method (1:trilinear, 2:cubic spline interpolation, 3:sinc interpolation)
                 output_suffix = '_ISO-'):
        """isovoxel data to desired resolution"""

        doc_vmr = self.bv.open_document(f'{data_dir}/{doc_vmr_fn}')

        #isovoxel
        print('\nIsovoxel {} to {} (framing: {})'.format(doc_vmr_fn, res, framing_cube))
        doc_vmr.transform_to_isovoxel(res,
                                      framing_cube,
                                      interpolation,
                                      f'{doc_vmr_fn.split(".")[0]}{output_suffix}{res}.vmr')

        return(self.bv.open_document('{}{}{}.vmr'.format(doc_vmr_fn,output_suffix,res)))   
        

    def coregister_bbr(self, data_dir, vmr_filename, fmr_filename):
        """do bbr coregistration, input directiory, vmr filename, and fmr filename
        the intitral coregistration will create the mash, subsequent will automatically use this mask"""
        
        # open up the isovoxeled uniden image
        doc_vmr = self.bv.open_document(vmr_filename)
        # do the coregistration
        print(doc_vmr.coregister_fmr_to_vmr_using_bbr(fmr_filename))
        print(doc_vmr.file_name)
        doc_vmr.close()
        

    def copy_mesh(self, data_dir, vmr_fn, masked_suffix='masked_'):
        """copy meshes for other filename, usefull if brainvoyager can use existing meshes (for example in bbr)
        vmr_fn just has to have the same dimensions, and is just uses so bv can actually load the meshes"""

        # file path lambda
        filepath = lambda filename: '{}/{}'.format(data_dir, filename)

        # set re search pattern 
        re_pattern = '(?=.*{})[a-zA-Z0-9._-]+.srf$'.format(masked_suffix)

        # search files
        dir_items = os.listdir(data_dir)
        dir_match = sorted([s for s in dir_items if re.search(re_pattern, s)])

        # open some vmr file so we can load mashes
        doc_vmr = self.bv.open_document(filepath(vmr_fn))

        # loop over mesh files
        for mesh_fn in dir_match:
        
            # open mesh
            doc_vmr.load_mesh(mesh_fn)
            mesh = doc_vmr.current_mesh
        
            # new filename + save
            mesh_new_fn = re.sub(masked_suffix, '', mesh.file_name)
            mesh.save_as(mesh_new_fn)

    def create_vtc(self,
                   data_dir,
                   vmr_fn,
                   fmr_fn,
                   vtc_fn,
                   ia_trf_fn,
                   fa_trf_fn,
                   use_bounding_box,           # whether or not to use bounding box
                   bounding_box_array = None,  # array of [[xfrom, xto], [yfrom, yto], [zfrom, zto]]
                   vtcspace = 1,               # create vtc in 1: native or 2: acpc space
                   acpc_trf_fn = None,         # if vtcspace is 2, give in acpc fn
                   extended_tal = False,       # use extened tal space for vtc creation (optional)
                   res_to_anat = 2,            # specify spatial resolution
                   interpolation_method = 2,   # interpolation method (0: nearest neighbor, 1: trilinear, 2: sinc)
                   bounding_box_int = 100,     # seperate background voxels from brain voxels
                   data_type = 2):             # 1: interger values, 2: float values
        """create vtc files in native or acpc space given settings"""
        
        # lambda to sellect file path
        filepath = lambda filename: '{}/{}'.format(data_dir, filename)
        doc_vmr = self.bv.open_document(vmr_fn)
        
        # check parameters
        doc_vmr.vtc_creation_extended_tal_space = extended_tal

        # if bounding box is true sellect it
        if use_bounding_box:
            # set values
            doc_vmr.vtc_creation_use_bounding_box = True
            doc_vmr.vtc_creation_bounding_box_from_x = bounding_box_array[0, 0]
            doc_vmr.vtc_creation_bounding_box_to_x   = bounding_box_array[0, 1]
            doc_vmr.vtc_creation_bounding_box_from_y = bounding_box_array[1, 0]
            doc_vmr.vtc_creation_bounding_box_to_y   = bounding_box_array[1, 1]
            doc_vmr.vtc_creation_bounding_box_from_z = bounding_box_array[2, 0]
            doc_vmr.vtc_creation_bounding_box_to_z   = bounding_box_array[2, 1]
        else:
            doc_vmr.vtc_creation_use_bounding_box = False

        # do the vtc creation in desired space
        if vtcspace == 1:
            doc_vmr.create_vtc_in_native_space(fmr_fn,
                                               ia_trf_fn,
                                               fa_trf_fn,
                                               vtc_fn,
                                               res_to_anat,
                                               interpolation_method,
                                               bounding_box_int,
                                               data_type)
        elif vtcspace == 2:
            acpc_trf_fn = filepath(acpc_trf_fn)
            doc_vmr.create_vtc_in_acpc_space(fmr_file,
                                             ia_trf_fn,
                                             fa_trf_fn,
                                             acpc_trf_fn,
                                             vtc_fn,
                                             res_to_anat,
                                             interpolation_method,
                                             bounding_box_int,
                                             data_type)

        return doc_vmr

    # nordic specific stuff
    def extract_files_gzip(self, filelist):
        new_filenames = []
        for file in filelist:
            extracted_file = file.replace('.gz', '')
            new_filenames.append(extracted_file)
            with gzip.open(file, 'rb') as f_in:
                with open(extracted_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return new_filenames

    def run_nordic(self, condition_path):
        os.chdir(condition_path)
        cmd = ['matlab', '-nodisplay', '-nosplash', '-r', f"run('{condition_path}RunNordic_Github_v2.m');exit;"]
        pprint(subprocess.check_output(cmd))

    # prt specific routines

    def create_log_dict(self, log_path, filename):

        with open(log_path + filename, 'r') as f:
            log = [np.array(list(filter(None, l.strip('\n').split('\t')))) for l in f.readlines()]
        
        keys = log[0]
        values = log[2:]

        log_data = {k:[] for k in keys}
        log_data['vol'] = []

        datetime_shift = datetime.strptime(values[0][-2], '%H:%M:%S.%f')

        for line in values:
            # discard test stimuli
            if line[0] == 'x': continue

            if len(line) == 7:
                trial, event, stim, start_time, duration, relative, volume = line
                exp = ''
            elif len(line) == 8:
                trial, event, stim, exp, start_time, duration, relative, volume = line

            log_data['Trial'].append(trial)
            log_data['Event'].append(event)
            log_data['Stim'].append(stim)
            log_data['Exp'].append(exp)
            log_data['Start Time'].append(start_time)
            log_data['Duration'].append(duration)
            relative_datetime = datetime.strptime(relative, '%H:%M:%S.%f')

            fixed_datetime = relative_datetime - datetime_shift
            fixed_datetime= float(f'{fixed_datetime.seconds}.{fixed_datetime.microseconds}')
            log_data['Relative'].append(fixed_datetime)

            log_data['vol'].append(int(fixed_datetime//2.5) + 1)

        return log_data

    def create_prt_data(self, log_data, conditions):
        sound_events = np.where(np.array(log_data['Event']) == 'Sound')[0]

        prt_data = {condition:[] for condition in conditions}

        for sound_idx in sound_events:
            sound = log_data["Stim"][sound_idx]
            exp = log_data["Exp"][sound_idx-1]

            # WhatOff
            if 'random' in log_data['Stim'][sound_idx-1]:
                condition = 'WhatOff_'
                if sound == 'omis':
                    condition += 'Omission'
                elif sound == 'pa':
                    condition += 'PA'
                elif sound == 'ga':
                    condition += 'GA'

            else:
                condition = 'WhatOn_'
                if sound == exp == 'pa':
                    condition += 'Congruent_PA'
                elif sound == exp == 'ga':
                    condition += 'Congruent_GA'
                elif sound == 'pa' and exp == 'ga':
                    condition += 'Incongruent_GA'
                elif sound == 'ga' and exp == 'pa':
                    condition += 'Incongruent_PA'
                elif sound == 'omis' and exp == 'pa':
                    condition += 'Omission_PA'
                elif sound == 'omis' and exp == 'ga':
                    condition += 'Omission_GA'

            end = log_data['vol'][sound_idx]
            
            # find start
            start_idx = sound_idx-1
            while log_data['Stim'][start_idx] != 'dummy':
                start_idx -= 1
            start = log_data['vol'][start_idx]

            trial = log_data['Trial'][sound_idx]
            # print(f'trial: {trial} \t expected: {exp} \t sound: {sound} \t condition: {condition}\t\t start: {start}\t\t end: {end}')
            prt_data[condition].append({'trial': trial, 'start': start, 'end': end})

        # add motor response entries
        volumes = np.array(log_data['vol'])[np.where((np.array(log_data['Event']) == 'Action') & 
                                                ((np.array(log_data['Stim']) == 'ga') | 
                                                 (np.array(log_data['Stim']) == 'pa'))
                                               )]
        trials = np.array(log_data['Trial'])[np.where((np.array(log_data['Event']) == 'Action') & 
            ((np.array(log_data['Stim']) == 'ga') | 
            (np.array(log_data['Stim']) == 'pa')))]

        for vol_idx, vol in enumerate(volumes):
            prt_data['MotorResponse'].append({'trial': trials[vol_idx], 'start': vol, 'end': vol})

        return prt_data

    def create_prt_data_split(self, log_data, conditions):
        sound_events = np.where(np.array(log_data['Event']) == 'Sound')[0]

        prt_data = {condition:[] for condition in conditions}

        for sound_idx in sound_events:
            sound = log_data["Stim"][sound_idx]
            exp = log_data["Exp"][sound_idx-1]

            # WhatOff
            if 'random' in log_data['Stim'][sound_idx-1]:
                condition = 'WhatOff_'
                if sound == 'omis':
                    condition += 'Omission'
                elif sound == 'pa':
                    condition += 'PA'
                elif sound == 'ga':
                    condition += 'GA'

            # WhatOn
            else:
                condition = 'WhatOn_'
                if sound == exp == 'pa':
                    condition += 'Congruent_PA'
                elif sound == exp == 'ga':
                    condition += 'Congruent_GA'
                elif sound == 'pa' and exp == 'ga':
                    condition += 'Incongruent_GA'
                elif sound == 'ga' and exp == 'pa':
                    condition += 'Incongruent_PA'
                elif sound == 'omis' and exp == 'pa':
                    condition += 'Omission_PA'
                elif sound == 'omis' and exp == 'ga':
                    condition += 'Omission_GA'

            end = log_data['vol'][sound_idx]
            
            # find start
            start_idx = sound_idx-1
            while log_data['Stim'][start_idx] != 'dummy':
                start_idx -= 1
            start = log_data['vol'][start_idx]

            trial = log_data['Trial'][sound_idx]
            # print(f'trial: {trial} \t expected: {exp} \t sound: {sound} \t condition: {condition}\t\t start: {start}\t\t end: {end}')
            prt_data[condition].append({'trial': trial, 'start': end, 'end': end})

            # add pictures
            if 'WhatOn' in condition:
                prt_data['WhatOn_Picture'].append({'trial': trial, 'start': start, 'end': start})
            elif 'WhatOff' in condition:
                prt_data['WhatOff_Picture'].append({'trial': trial, 'start': start, 'end': start})

        # add motor response entries
        volumes = np.array(log_data['vol'])[np.where((np.array(log_data['Event']) == 'Action') & 
                                                ((np.array(log_data['Stim']) == 'ga') | 
                                                 (np.array(log_data['Stim']) == 'pa'))
                                               )]
        trials = np.array(log_data['Trial'])[np.where((np.array(log_data['Event']) == 'Action') & 
            ((np.array(log_data['Stim']) == 'ga') | 
            (np.array(log_data['Stim']) == 'pa')))]

        for vol_idx, vol in enumerate(volumes):
            prt_data['MotorResponse'].append({'trial': trials[vol_idx], 'start': vol, 'end': vol})

        return prt_data


    def create_prt_data_trialsplit(self, log_data, conditions):
        sound_events = np.where(np.array(log_data['Event']) == 'Sound')[0]
        prt_data = {'WhatOn_Picture':[], 'WhatOff_Picture':[], 'MotorResponse':[]}

        for trial_idx, sound_idx in enumerate(sound_events):
            sound = log_data["Stim"][sound_idx]
            exp = log_data["Exp"][sound_idx-1]

            # WhatOff
            if 'random' in log_data['Stim'][sound_idx-1]:
                condition = 'WhatOff_'
                if sound == 'omis':
                    condition += 'Omission'
                elif sound == 'pa':
                    condition += 'PA'
                elif sound == 'ga':
                    condition += 'GA'

            # WhatOn
            else:
                condition = 'WhatOn_'
                if sound == exp == 'pa':
                    condition += 'Congruent_PA'
                elif sound == exp == 'ga':
                    condition += 'Congruent_GA'
                elif sound == 'pa' and exp == 'ga':
                    condition += 'Incongruent_GA'
                elif sound == 'ga' and exp == 'pa':
                    condition += 'Incongruent_PA'
                elif sound == 'omis' and exp == 'pa':
                    condition += 'Omission_PA'
                elif sound == 'omis' and exp == 'ga':
                    condition += 'Omission_GA'
            condition += f'_trial_{trial_idx}'

            end = log_data['vol'][sound_idx]
            
            # find start
            start_idx = sound_idx-1
            while log_data['Stim'][start_idx] != 'dummy':
                start_idx -= 1
            start = log_data['vol'][start_idx]

            trial = log_data['Trial'][sound_idx]
            # print(f'trial: {trial} \t expected: {exp} \t sound: {sound} \t condition: {condition}\t\t start: {start}\t\t end: {end}')
            prt_data.update({condition: [{'trial': trial, 'start': end, 'end': end}]})

            # add pictures
            if 'WhatOn' in condition:
                prt_data['WhatOn_Picture'].append({'trial': trial, 'start': start, 'end': start})
            elif 'WhatOff' in condition:
                prt_data['WhatOff_Picture'].append({'trial': trial, 'start': start, 'end': start})

        # add motor response entries
        volumes = np.array(log_data['vol'])[np.where((np.array(log_data['Event']) == 'Action') & 
                                                ((np.array(log_data['Stim']) == 'ga') | 
                                                 (np.array(log_data['Stim']) == 'pa'))
                                               )]
        trials = np.array(log_data['Trial'])[np.where((np.array(log_data['Event']) == 'Action') & 
            ((np.array(log_data['Stim']) == 'ga') | 
            (np.array(log_data['Stim']) == 'pa')))]

        for vol_idx, vol in enumerate(volumes):
            prt_data['MotorResponse'].append({'trial': trials[vol_idx], 'start': vol, 'end': vol})


        return {k: v for k, v in prt_data.items() if v != []}  # remove empty conditions


    def create_prt_content(self, prt_data, conditions, colors):
        header = '''FileVersion:        2

ResolutionOfTime:   Volumes

Experiment:         Associative learning

BackgroundColor:    0 0 0
TextColor:          255 255 255
TimeCourseColor:    255 255 255
TimeCourseThick:    3
ReferenceFuncColor: 0 0 80
ReferenceFuncThick: 3

'''
        content = f'NrOfConditions:     {len(conditions)}\n\n'
        ncols = len(colors)
        for color_idx, condition in enumerate(prt_data.keys()):
            content += f'{condition}\n'
            content += f'{len(prt_data[condition])}\n'
            for trial in prt_data[condition]:
                content += f'\t{trial["start"]}   {trial["end"]}\n'
            content += f'Color: {colors[color_idx%ncols][0]} {colors[color_idx%ncols][1]} {colors[color_idx%ncols][2]}\n\n'


        return header + content

#     def create_prt_content_split_av(self, prt_data, conditions, colors):
#         header = '''FileVersion:        2

# ResolutionOfTime:   Volumes

# Experiment:         Associative learning

# BackgroundColor:    0 0 0
# TextColor:          255 255 255
# TimeCourseColor:    255 255 255
# TimeCourseThick:    3
# ReferenceFuncColor: 0 0 80
# ReferenceFuncThick: 3

# '''

#         content = f'NrOfConditions:     {len(conditions)}\n\n'
#         for color_idx, condition in enumerate(prt_data.keys()):
#             content += f'{condition}\n'
#             content += f'{len(prt_data[condition])}\n'
#             for trial in prt_data[condition]:
#                 content += f'\t{trial["start"]}   {trial["end"]}\n'
#             content += f'Color: {colors[color_idx][0]} {colors[color_idx][1]} {colors[color_idx][2]}\n\n'


#         return header + content

    # pulse correctness
    def verify_TR_correctness(self, log_data, tr_list_filename, verbose=False):
        with open(tr_list_filename, 'rb') as f: tr_list = pkl.load(f)

        # durations = np.array(log_data['Duration'])[np.where(np.array(log_data['Stim']) == 'PULSE')]
        pulses = np.array(log_data['Stim'])[np.where((np.array(log_data['Event']) == 'Action') | (np.array(log_data['Event']) == 'Sound'))]

        count_dups = [sum(1 for _ in group) for _, group in groupby(pulses)]
        count_dups = [elem for elem in count_dups if elem != 1]

        tr_counts = {'TR_list_after_sound':[], 'TR_list_after_choice':[]}
        pos = 'TR_list_after_sound'

        for c in count_dups:
            tr_counts[pos].append(c)
            if pos == 'TR_list_after_sound':
                pos = 'TR_list_after_choice'
            else:
                pos = 'TR_list_after_sound'

        if tr_counts == tr_list:
            if verbose: print('Experimental TR verfication was successfull!')
            return True
        print(f'Warning, logged TR\'s don\'t match with data at {tr_list_filename}.')

        if tr_counts['TR_list_after_sound'] != tr_list['TR_list_after_sound']:
            print('error is in TR_list_after_sound')
            print(f'\ttr log: {tr_counts["TR_list_after_sound"]}')
            print(f'\ttr lst: {tr_list["TR_list_after_sound"]}')
        if tr_counts['TR_list_after_choice'] != tr_list['TR_list_after_choice']:
            print('error is in TR_list_after_choice')
            print(f'\ttr log: {tr_counts["TR_list_after_choice"]}')
            print(f'\ttr lst: {tr_list["TR_list_after_choice"]}')
            deviations = np.nonzero(np.abs(list(map(int.__sub__, tr_counts["TR_list_after_choice"], tr_list["TR_list_after_choice"]))))[0]
            for dev in deviations:
                print(f'Deviation at index: {dev}, check for "none" choices!')

        return False

    # stim correctness
    def verify_stim_correctness(sefl, log_data, stim_list_filename, verbose=False):
        with open(stim_list_filename, 'rb') as f: stim_list = pkl.load(f)

        pictures = np.array(log_data['Stim'])[np.where(np.array(log_data['Event']) == 'Picture')]
        exp = np.array(log_data['Exp'])[np.where(np.array(log_data['Event']) == 'Picture')]

        pics_with_exp = []
        for i,e in enumerate(exp):
            if e != '':
                s = pictures[i] + '=' + exp[i]
            else:
                s = pictures[i]
            pics_with_exp.append(s)

        pics_with_exp = np.array(pics_with_exp).reshape(len(pics_with_exp)//3, 3)

        sounds = np.array(log_data['Stim'])[np.where(np.array(log_data['Event']) == 'Sound')]

        stims = [list(np.append(p, sounds[i])) for i,p in enumerate(pics_with_exp)]

        if stims == stim_list:
            if verbose: print('Experimental stimulus verfication was successfull!')
            return True
        print('Warning, logged stimuli don\'t match with data')
        return False
