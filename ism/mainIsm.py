
# MAIN FUNCTION TO CALL THE ISM MODULE

from ism.src.ism import ism

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r'C:\\Users\\Usuario\\Documents\\GitHub\\EODP\\auxiliary'

#for all others
#indir = r"C:\\Users\\Usuario\\Documents\\GitHub\\EODP\\EODP_TER_2021\\EODP-TS-ISM\\input\\gradient_alt100_act150" # small scene
#outdir = r"C:\\Users\\Usuario\\Documents\\GitHub\\EODP\\EODP_TER_2021\\EODP-TS-ISM\\outputBoris"

# for E2E
indir = r"C:/Users/Usuario/Documents/GitHub/EODP/EODP_TER_2021/EODP-TS-E2E/sgm_out" # small scene
outdir = r"C:/Users/Usuario/Documents/GitHub/EODP/EODP_TER_2021/EODP-TS-E2E/output_Boris"

# Initialise the ISM
myIsm = ism(auxdir, indir, outdir)
myIsm.processModule()
