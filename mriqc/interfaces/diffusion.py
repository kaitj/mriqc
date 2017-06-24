#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author:
# @Date:
# @Email:
""" Nipype interfaces to support diffusion workflow """ 

from __future__ import print_function, division, absolute_import, unicode_literals
from os import path as op
import numpy as np
import nibabel as nb
from nilearn.signal import clean
from builtins import zip

from niworkflows.nipype.interfaces.base import (
	traits, TraitedSpec, OutputMultiPath, File)
from niworkflows.nipype import logging
from niworkflows.nipype.utils import NUMPY_MMAP
from niworkflows.interfaces.base import SimpleInterface

class ExtractB0InputSpec(BaseInterfaceInputSpec):
	in_dwi = File(exists=True, mandatory=True, desc='DWI file to extract b0 images')
	in_bval = File(exists=True, mandatory=True, desc='b-values file')
	
class ExtractB0OutputSpec(TraitedSpec):
	ref_img = File(exists=True, desc='First B0 image in dwi data to be used as reference')
	out_imgs = OutputMultiPath(File(exists=True, desc='All other B0 images'))
	
class ExtractB0(SimpleInterface):
	"""
	Computes and extracts b0 images from DWI data outputting the first b0 image as the reference
	"""
	input_spec = ExtractB0InputSpec
	output_spec = ExtractB0OutputSpec
	
	def _run_interface(self, runtime):
		imgs = np.array(nb.four_to_three(nb.load(self.inputs.in_dwi, mmap=NUMPY_MMAP)))
		bval = np.loadtxt(self.inputs.in_bval)
		max_b = 10.0
		
		index = np.argwhere(bval <= max_b).flatten().tolist()
		
		b0s = [im.get_data().astype(np.float32)
			for im in imgs[index]]
			
		hdr = imgs[0].header.copy()
		hdr.set_data_shape(b0s[0].shape)
		hdr.set_xyzt_units('mm')
		hdr.set_data_dtype(np.uint8)
		
		# Output initial b0
		self._results['ref_img'] = op.abspath('dwib0_0.nii.gz')
		nb.Nifti1Image(b0s[0], imgs[0].affine, hdr).to_filename(
			self._results['ref_img'])
		
		# Output other b0s
		out_imgs = []
		count = 1
		for b0 in b0s[1:]:
			filename.append(op.abspath('dwib0_%03i.nii.gz' % count))
			nb.Nifti1Image(b0, imgs[0].affine, hdr).to_filename(
				filename[count-1])
			count += 1 
			
		self._results['out_imgs'] = filename
			
	return runtime