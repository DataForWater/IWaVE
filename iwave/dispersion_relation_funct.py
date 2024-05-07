import numpy as np

def dispersion_relation_funct(kx, ky, d, m, v_x, v_y):
  """calculated the frequency of gravity-capillary waves and turbulence-forced waves according to velocity v_x and v_y, depth d, and shear factor m."""
  
    g = 9.81  # gravity acceleration (m/s^2)
    surftens = 72.75e-03  # surface tension (N/m), ref. 20deg, from Vargaftik et al., 1983 - International tables of the surface tension of water, J.Phys.Chem.Ref.Data
    dens = 998.2  # water density (kg/m^3), ref. 20deg

    Kx, Ky = np.meshgrid(kx, ky)
    K = np.sqrt(Kx**2 + Ky**2) # wavenumber modulus

    # frequency of gravity-capillary waves
    
    #Kt_gw = np.where(K !=0, (Kx*v_x + Ky*v_y)*(1-(m/2)*np.tanh(K*d)/(K*d)) +\
    #    np.sqrt((g+surftens/dens*K**2)*K*np.tanh(K*d)+(m*(Kx*v_x + Ky*v_y)*np.tanh(K*d)/(2*K*d))**2), 0)
    
    # separate terms that go to Inf when K=0
    term1 = np.divide((m/2)*np.tanh(K*d), K*d, out=np.zeros_like(K), where=K!=0)
    term2 = np.divide((Kx*v_x + Ky*v_y)*np.tanh(K*d), 2*K*d, out=np.zeros_like(K), where=K!=0)

    Kt_gw = np.where(K != 0, (Kx*v_x + Ky*v_y)*(1-term1) +\
        np.sqrt((g+surftens/dens*K**2)*K*np.tanh(K*d)+term2**2), 0)

    # frequency of turbulence-generated waves
    Kt_turb = Kx*v_x + Ky*v_y

    # Kt_gw[K==0] = 0    # force to 0 at K=0
    # Kt_turb[K==0] = 0  # force to 0 at K=0

    return Kt_gw, Kt_turb
