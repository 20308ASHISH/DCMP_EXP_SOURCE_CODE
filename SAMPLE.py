# Importing necessary libraries
from pythtb import *
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

# Initializing the Wannier90 model by reading the data files
pgda=w90(r"PdGa_Ham2",r"wannier90")
my_model=pgda.model(min_hopping_norm=0.01) # only hoppings with norm greater than 0.01 are considered



def calculate_v(k_vec, dk=0.01):
    Hxp = my_model._gen_ham([k_vec[0]+dk, k_vec[1], k_vec[2]])
    Hyp = my_model._gen_ham([k_vec[0], k_vec[1]+dk, k_vec[2]])
    Hzp = my_model._gen_ham([k_vec[0], k_vec[1], k_vec[2]+dk])

    H = my_model._gen_ham([k_vec[0], k_vec[1], k_vec[2]])

    vx = (Hxp-H)/(dk)
    vy = (Hyp-H)/(dk)
    vz = (Hzp-H)/(dk)
    return np.array((vx, vy, vz))



### ___________________________________________________Define orbital angular momentum L_z matrix__________________________________________
# Lz for d orbitals (5x5 matrix for Pd atoms)
Lz_d = np.array([
    [0,  0,    0,   0,   0],
    [0,  0,  -1j,   0,   0],
    [0, 1j,    0,   0,   0],
    [0,  0,    0,   0, -2j],
    [0,  0,    0,  2j,   0]
], dtype=complex)

# Lz for p orbitals (3x3 matrix for Ga atoms)
Lz_p = np.array([
    [0,  0,    0],
    [0,  0,  -1j],
    [0, 1j,    0]
], dtype=complex)

#d_5
## Make 20*20
temp = np.zeros((5,5) , dtype = complex)
d_1010 = np.vstack((np.hstack((Lz_d,temp)), np.hstack((temp, Lz_d))))
temp1 = np.zeros((10,10), dtype = complex)
d_2020 = np.vstack((np.hstack((d_1010,temp1)), np.hstack((temp1, d_1010))))

#p_3
## Make 12*12
temp2 = np.zeros((3,3) , dtype = complex)
p_66 = np.vstack((np.hstack((Lz_p,temp2)), np.hstack((temp2, Lz_p))))
temp3 = np.zeros((6,6), dtype = complex)
p_1212 = np.vstack((np.hstack((p_66,temp3)), np.hstack((temp3, p_66))))


##Make the 32*32
ul_block = np.zeros((32, 32), dtype=complex)
ul_block[:20, :20] = d_2020       # top-left 20x20: d orbitals
ul_block[20:, 20:] = p_1212       # bottom-right 12x12: p orbitals

##Make the whole matrix
Lz_full = np.zeros((64, 64), dtype=complex)
# Upper left block: spin-up_up
Lz_full[0:32, 0:32] = ul_block
# Lower right block: spin-down_down
Lz_full[32:64, 32:64] = ul_block


# To get energy eigenvalues and corresponding OBC (Omega^z_xy) at a k-point
def solve_and_calculate_Omega_xyz(k_vec, eta=0.01):
    v = calculate_v(k_vec)                  # velocity using forward finite difference
    (eig_values, eig_vectors) = my_model.solve_one(k_vec, eig_vectors=True)
    
    # Calculating L_z
    Lz = np.zeros(eig_values.shape[0])
    
    # Calculating Omega^z_xy
    Omega_xyz = np.zeros(eig_values.shape[0])
    for n in range(eig_values.shape[0]):
        temp = 0
        for m in range(eig_values.shape[0]):
            if m==n:
                continue
            temp -= (eig_vectors[n].conjugate() @ Lz_full @ eig_vectors[n] + eig_vectors[m].conjugate() @ Lz_full @ eig_vectors[m])*(eig_vectors[n].conjugate().dot(v[0]).dot(eig_vectors[m])*eig_vectors[m].conjugate().dot(v[1]).dot(eig_vectors[n]))/(eig_values[n]-eig_values[m]+complex(0,eta))**2
        Omega_xyz[n] = np.imag(temp/2)
    return (eig_values, Omega_xyz)


## Function to calculate OHC at a k-point

# To get energy eigenvalues and corresponding OBC (Omega^z_xy) at a k-point
def calculate_ohc_k_point(eig_values, Omega_xyz, mu):
    # Calculate the Fermi-Dirac distribution
    beta = 1 / 0.26
    fermi_dirac = 1 / (np.exp((eig_values  - 6.87 + mu)* beta) + 1)

    # print("Fermi-Dirac distribution:", fermi_dirac)
    # Calculate the OHC
    ohc = np.sum(Omega_xyz * fermi_dirac)
    
    return ohc


def compute_ohc_for_k(args):
    k_point, mu_val = args
    eigvals, Omega = solve_and_calculate_Omega_xyz(k_point)
    return calculate_ohc_k_point(eigvals, Omega, mu_val)  # You must pass mu if it's used inside

if __name__ == "__main__":
    tm_st = time.time()
    
    N = 12
    ohc_list = []
    grid = np.linspace(-0.5, 0.5, N)
    k_points = np.array([[x, y, z] for x in grid for y in grid for z in grid])
    mu_values = np.linspace(-0.1,0.1,20)

    for mu in mu_values:
        # Create argument list: list of (k_point, mu) tuples
        args_list = [(k, mu) for k in k_points]

        print(f"Running for mu = {mu:.2f} ...")
        with Pool(processes=6) as pool:
            results = pool.map(compute_ohc_for_k, args_list)

        # Final sum and scaling
        ohc = sum(results) / (2 * np.pi)**3 / N**3
        ohc_list.append([mu, ohc])
        print(f"OHC for mu = {mu:.2f} is {ohc:.6f}")

    tm_end = time.time()
    print(f"Total time taken: {tm_end - tm_st:.2f} seconds")

    ohc_list = np.array(ohc_list)

np.savetxt("ohc_calculation_data.csv", ohc_list, delimiter=",", fmt="%0.16f")