""" Defaults for using nipype
"""
import nipype.interfaces.matlab as nim
# If you needed to set the default matlab command above
nim.MatlabCommand.set_default_matlab_cmd('/Applications/MATLAB_R2014a.app/bin/matlab')
# If you needed to at the SPM path above
nim.MatlabCommand.set_default_paths('/Users/mb312/spm')
