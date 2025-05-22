
from pythtb import  *    #import TB model class
import matplotlib.pyplot as plt
import pickle
import ray
import time
import matplotlib.pyplot as plt


### read your model from wannier90 files
pdga=w90(r"./PdGa_Ham2",r"wannier90")

fermi_ev = 0.00      #Use correct fermi level according to your system
my_model=pdga.model(zero_energy=fermi_ev, min_hopping_norm=0.001) 




## save the model in python pickle format, this will save the time in next time
## you can load  the model from the pickle file
## and use it directly
## this is not necessary if you are using the model for the first time
## DO IT ONCE ONLY


#pickle.dump(my_model,open("my_model.pkl","wb"))
#my_model=pickle.load(open("my_model.pkl","rb"))


#%% K-path for model
path=[[0.5000,  0.0000,   0.0000], [0.0000,  0.0000,   0.0000], [0.0000,  0.5000,   0.0000],
      [0.5000,  0.5000,   0.0000], [0.0000,  0.0000,   0.0000]]
k_label=(r'$X$', r'$\Gamma$',r'$Y$', r'$S$', r'$\Gamma$')
(k_vec,k_dist,k_node)=my_model.k_path(path,80)



#%% series computaion
start_time = time.time()
evals=my_model.solve_all(k_vec)
duration = time.time() - start_time
print('time==', duration)  

#%%
fig, ax = plt.subplots()
for i in range(evals.shape[0]):
    ax.plot(k_dist,evals[i]+1.8114,"k-")
for n in range(len(k_node)):
    ax.axvline(x=k_node[n],linewidth=0.5, color='k')
ax.set_xlabel("Path in k-space")
ax.set_ylabel("Band energy (eV)")
ax.set_xlim(k_dist[0],k_dist[-1])
ax.set_ylim(-1,1)
ax.set_xticks(k_node)
ax.set_xticklabels(k_label)
fig.tight_layout()

#%%###################################################################
################# PARALLEL COMPUTATION OF BANDS #################
######################################################################
## DEFINING FUCNTION 
@ray.remote
def get_eigenvalue(my_model, k_vec):
    evals_ray=my_model.solve_one(k_vec)
    return evals_ray

############ USING RAY TO COMPUTE THE EIGENVALUES #####################
ray.init(num_cpus=12)    # this will use 12 cpus, change according to your system
model_id= ray.put(my_model)                  #assign a id to the model
start_time = time.time()
k1=k_vec
idx=[]
results=np.array([])

for k in k1:
    idx.append(get_eigenvalue.remote(model_id,k))

results = ray.get(idx)
results=np.transpose(results)

duration = time.time() - start_time
print('time==', duration)
ray.shutdown()
#%%######################################################################
######################## PLOT THE BAND STRUCTURE ########################
#########################################################################
fig, ax = plt.subplots()
for i in range(results.shape[0]):
    ax.plot(k_dist,results[i],"k-")
for n in range(len(k_node)):
    ax.axvline(x=k_node[n],linewidth=0.5, color='k')
ax.set_xlabel("Path in k-space")
ax.set_ylabel("Band energy (eV)")
ax.set_xlim(k_dist[0],k_dist[-1])
ax.set_ylim(-1,1)
ax.set_xticks(k_node)
ax.set_xticklabels(k_label)
fig.tight_layout()



#%%
#1. Compare the time taken for the serial and parallel computation of the eigenvalues.
