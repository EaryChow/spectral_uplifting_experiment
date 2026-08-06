import numpy
import colour

numpy.set_printoptions(suppress=True, precision=128)

source_rgb_space = 'ITU-R BT.709'
reconstruction_method = 'Jakob 2019'
rgb = numpy.array([1.0, 1.0, 1.0])
achromatic_name = 'E'
print(achromatic_name)
achromatic = colour.SDS_ILLUMINANTS[achromatic_name]
achromatic_xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][achromatic_name]

rgb_wp_xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][colour.RGB_COLOURSPACES[source_rgb_space].whitepoint_name]
rgb_wp_XYZ = colour.xy_to_XYZ(rgb_wp_xy)
e_wp_XYZ = colour.xy_to_XYZ(colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['E'])

adaptation_matrix = colour.adaptation.matrix_chromatic_adaptation_VonKries(
    rgb_wp_XYZ, e_wp_XYZ, transform='CAT02')

source_XYZ = colour.RGB_to_XYZ(rgb, colourspace=source_rgb_space)
source_XYZ = numpy.dot(source_XYZ, adaptation_matrix.T)

cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
spectra = colour.XYZ_to_sd(source_XYZ, illuminant=achromatic, method=reconstruction_method, cmfs=cmfs)

if numpy.any(spectra.values < 0):
    print("Warning: The reconstructed spectra contains negative energy values.")

inverse_adaptation_matrix = numpy.linalg.inv(adaptation_matrix)
reconstructed_xy = colour.XYZ_to_xyY(colour.sd_to_XYZ(spectra, cmfs=cmfs, illuminant=colour.SDS_ILLUMINANTS['E']))
reconstructed_xy_post_adaptation = colour.XYZ_to_xyY(numpy.dot(colour.sd_to_XYZ(spectra, cmfs=cmfs, illuminant=colour.SDS_ILLUMINANTS['E']), inverse_adaptation_matrix.T))

# output_filename = f'{source_rgb_space}_({rgb[0]}_{rgb[1]}_{rgb[2]})_{reconstruction_method}.csv'

# # Variable to control the precision of the written numbers
# precision = 6

# # Manually write CSV content to prevent scientific notation
# with open(output_filename, 'w') as f:
#     f.write("wavelength,power\n")
#     for i in range(len(spectra.wavelengths)):
#         f.write(f"{spectra.wavelengths[i]:.1f},{spectra.values[i]:.{precision}f}\n")

source_xyY = colour.XYZ_to_xyY(colour.RGB_to_XYZ(rgb, colourspace=source_rgb_space))
print_precision = 3
print(f'uplifting done! original xyY is {numpy.round(source_xyY, print_precision).tolist()}, converted spectra xyY pre-adaptation is {numpy.round(reconstructed_xy, print_precision).tolist()}')
print(f'converted spectra xyY post-adaptation is {numpy.round(reconstructed_xy_post_adaptation, print_precision).tolist()}')