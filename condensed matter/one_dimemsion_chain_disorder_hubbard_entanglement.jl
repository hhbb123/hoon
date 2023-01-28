using LinearAlgebra
using Plots
using DataFrames
using CSV
using Statistics

# 1d-hubbard chain
function Hamiltonian_up(t,U,n_down_set,k)
  Hamiltonian_up_matrix=zeros(length(n_down_set),length(n_down_set))
  for i in range(2,length(n_down_set)-1)
    Hamiltonian_up_matrix[i,i+1]=-t*cos(k)
    Hamiltonian_up_matrix[i,i-1]=-t*cos(k)
    Hamiltonian_up_matrix[i+1,i]=-t*cos(k)
    Hamiltonian_up_matrix[i-1,i]=-t*cos(k)
    Hamiltonian_up_matrix[i,i]=U*n_down_set[i]-0.5*U
  end
  Hamiltonian_up_matrix[length(n_down_set),1]=-t*cos(k)
  Hamiltonian_up_matrix[1,length(n_down_set)]=-t*cos(k)
  Hamiltonian_up_matrix[length(n_down_set)-1,length(n_down_set)]=-t*cos(k)
  Hamiltonian_up_matrix[length(n_down_set),length(n_down_set)-1]=-t*cos(k)
  Hamiltonian_up_matrix[length(n_down_set),length(n_down_set)]=U*n_down_set[length(n_down_set)]-0.5*U

  return Hamiltonian_up_matrix
end

function Hamiltonian_down(t,U,n_up_set,k)
  Hamiltonian_down_matrix=zeros(length(n_up_set),length(n_up_set))
  for i in range(2,length(n_up_set)-1)
    Hamiltonian_down_matrix[i,i+1]=-t*cos(k)
    Hamiltonian_down_matrix[i,i-1]=-t*cos(k)
    Hamiltonian_down_matrix[i+1,i]=-t*cos(k)
    Hamiltonian_down_matrix[i-1,i]=-t*cos(k)
    Hamiltonian_down_matrix[i,i]=U*n_up_set[i]-0.5*U
  end
  Hamiltonian_down_matrix[length(n_up_set),1]=-t*cos(k)
  Hamiltonian_down_matrix[1,length(n_up_set)]=-t*cos(k)
  Hamiltonian_down_matrix[length(n_up_set)-1,length(n_up_set)]=-t*cos(k)
  Hamiltonian_down_matrix[length(n_up_set),length(n_up_set)-1]=-t*cos(k)
  Hamiltonian_down_matrix[length(n_up_set),length(n_up_set)]=U*n_up_set[length(n_up_set)]-0.5*U

  return Hamiltonian_down_matrix
end

function correlation_matrix(W,l,s)
  C_m=zeros(l,l)
  for i in range(1,l)
    for j in range(1,l)
      for m in range(1,s)
        C_m[i,j] += conj(W[i,m])*W[j,m]
      end
    end
  end
  return C_m
end

#entanglement entropy
function entropy(C_m,l)
  C_m_l=zeros(l,l)
  for i in range(1,l)
    for j in range(1,l)
      C_m_l[i,j]=C_m[i,j]
    end
  end
  eta,cf=eigen(C_m_l)
  S=0
  for i in range(1,length(eta))
    if eta[i] <1 && eta[i] > 0
      S += -eta[i]*log(eta[i])-(1-eta[i])*log(1-eta[i])
    end
  end
  return S
end


function d2N(A,l,s) #find number fluctuation function
  A1 = *(A[1:l,1:s],conj(transpose(A[1:l,1:s])))
  A2 = *(conj(A[1:l,s+1:2*s]),transpose(A[1:l,s+1:2*s]))
  return sum(A1.*A2)
end


function disorder(N,V,n) # N=lattice number, V=disorder interaction
  D_matrix=zeros(N,N)
  for i=1:N
    if n>rand(Float64)
      D_matrix[i,i]=V*rand(Float64)*((-1)^rand(Int))
    end
  end
  return D_matrix
end

function HF(N_H,S_H,t,U,Di) #Hartree-fock approximation

  n_up_set=zeros(N_H)
  n_down_set=zeros(N_H)

  for i in range(1,S_H) #initial value
    n_up_set[2*i]= 0.5
    n_down_set[2*i]=0.5
    n_up_set[2*i-1]=0.5
    n_down_set[2*i-1]=0.5
  end

  global E_prev=1
  global E=1000

  while abs(E_prev-E)>1e-16
    global V_up,F_up=eigen(Hamiltonian_up(t,U,n_down_set,0)+Di)

    for j in range(1,N_H)
      n_up_set[j]=0
    end

    for j in range(1,N_H)
      for m in range(1,S_H)  #half-filling
        n_up_set[j] += conj(F_up[j,m])*F_up[j,m]
      end
    end

    global V_down,F_down=eigen(Hamiltonian_up(t,U,n_up_set,0)+Di)

    for j in range(1,N_H)
      n_down_set[j]=0
    end

    for j in range(1,N_H)
      for m in range(1,S_H)  #half-filling
        n_down_set[j] += conj(F_down[j,m])*F_down[j,m]
      end
    end

    global e_up=0
    global e_down=0
    global e_interaction=0
    for j in range(1,S_H)
      global e_up += V_up[j]
    end
    for j in range(1,S_H)
      global e_down += V_down[j]
    end
    for j in range(1,N_H)
      global e_interaction += U*n_up_set[j]*n_down_set[j]
    end

    global E_prev=E
    global E=e_up+e_down-e_interaction
  end
  return F_up,V_up, n_up_set
end

N_H=256 #lattice number
S_H=128 # spin-up number

# value
t=1.0 #hopping 
U=2 #interaction


re=50 #number of sample

S_m=zeros(Int(S_H),re)
D_m=zeros(Int(S_H),re)
N_m=zeros(N_H,re)
Energy_spec=[]
for i=1:re
  F,V,N_up=HF(N_H,S_H,t,U,disorder(N_H,1,1)) # V=1.0
  C_m_512=correlation_matrix(F,N_H,S_H)
  ent=[]
  dN=[]
  for l=1:S_H
    append!(ent,entropy(C_m_512,l))
    append!(dN,d2N(F,l,S_H))
  end

  append!(Energy_spec,V)


  for j=1:Int(S_H)
    S_m[j,i]=ent[j]
    D_m[j,i]=dN[j]
  end

  for j=1:N_H
    N_m[j,i]=N_up[j]
  end

end

S=zeros(Int(S_H))
D=zeros(Int(S_H))
std_S=zeros(Int(S_H))
std_D=zeros(Int(S_H))

N_up_r=zeros(N_H)

for i=1:Int(S_H)
  std_S[i]=std(S_m[i,:])
  std_D[i]=std(D_m[i,:])
  for j=1:re
    S[i] += S_m[i,j]/re
    D[i] += D_m[i,j]/re
  end
end

for i=1:N_H
  for j=1:re
    N_up_r[i] += N_m[i,j]/re
  end
end

L=[]
L_log=[]
for i=1:S_H
  append!(L,i)
  append!(L_log,log(i))
end


# make CSV file
parameter=DataFrame(size_log=L_log, entropy=S, number_fluctuation=D,entropy_error=std_S,fluctuation_error=std_D)

spectrum=DataFrame(spectrum=Energy_spec)

particle_n=DataFrame(particle_number=N_up_r)

CSV.write("length-entropy-numberfluctuation.csv",parameter)
CSV.write("DOS.csv",spectrum)
CSV.write("density-distribution.csv",particle_n)
