import numpy as np
import math

def ray_param(v, theta):
    '''
    Calculates the ray parameter p
    
    Usage:
    ------
        p = ray_param(v, theta)
    
    Inputs:
    -------
            v = interval velocity
        theta = incidence angle of ray (degrees)
    
    Output:
    -------
        p = ray parameter (i.e. sin(theta)/v )
    '''

    
#    # Cast inputs to floats
#    theta = float(theta)
#    v = float(v)
    
    p = math.sin(math.radians(theta))/v # ray parameter calculation
    
    return p

def rc_zoep(vp1, vs1, rho1, vp2, vs2, rho2, theta1):
    '''
    Reflection & Transmission coefficients calculated using full Zoeppritz
    equations.
    
    Usage:
    ------
    R = rc_zoep(vp1, vs1, rho1, vp2, vs2, rho2, theta1)
    
    Reference:
    ----------
    The Rock Physics Handbook, Dvorkin et al.
    '''
#    
#     Cast inputs to floats
#    vp1  = float(vp1)
#    vp2  = float(vp2)
#    vs1  = float(vs1)
#    vs2  = float(vs2)
#    rho1 = float(rho1)
#    rho2 = float(rho2)
#    theta1 = float(theta1)
    
    # Calculate reflection & transmission angles
    theta1 = math.radians(theta1)   # Convert theta1 to radians
    p      = ray_param(vp1, math.degrees(theta1)) # Ray parameter
    temp = min(1,max(p*vp2,-1))
    theta2 = math.asin(temp);      # Transmission angle of P-wave
    temp = min(1,max(p*vs1,-1))
    phi1   = math.asin(temp);      # Reflection angle of converted S-wave
    temp = min(1,max(p*vs2,-1))
    phi2   = math.asin(temp);      # Transmission angle of converted S-wave
    
    # Matrix form of Zoeppritz Equations... M & N are two of the matricies
    M = np.array([ \
        [-math.sin(theta1), -math.cos(phi1), math.sin(theta2), math.cos(phi2)],\
        [math.cos(theta1), -math.sin(phi1), math.cos(theta2), -math.sin(phi2)],\
        [2*rho1*vs1*math.sin(phi1)*math.cos(theta1), rho1*vs1*(1-2*math.sin(phi1)**2),\
            2*rho2*vs2*math.sin(phi2)*math.cos(theta2), rho2*vs2*(1-2*math.sin(phi2)**2)],\
        [-rho1*vp1*(1-2*math.sin(phi1)**2), rho1*vs1*math.sin(2*phi1), \
            rho2*vp2*(1-2*math.sin(phi2)**2), -rho2*vs2*math.sin(2*phi2)]
        ], dtype='float')
    
    N = np.array([ \
        [math.sin(theta1), math.cos(phi1), -math.sin(theta2), -math.cos(phi2)],\
        [math.cos(theta1), -math.sin(phi1), math.cos(theta2), -math.sin(phi2)],\
        [2*rho1*vs1*math.sin(phi1)*math.cos(theta1), rho1*vs1*(1-2*math.sin(phi1)**2),\
            2*rho2*vs2*math.sin(phi2)*math.cos(theta2), rho2*vs2*(1-2*math.sin(phi2)**2)],\
        [rho1*vp1*(1-2*math.sin(phi1)**2), -rho1*vs1*math.sin(2*phi1),\
            -rho2*vp2*(1-2*math.sin(phi2)**2), rho2*vs2*math.sin(2*phi2)]\
        ], dtype='float')
    
    # This is the important step, calculating coefficients for all modes and rays
    R = np.dot(np.linalg.inv(M), N);
    
    return R


def simulate_np(vp, vs, den, angle):
    
#    vp = np.reshape(vp,[-1,])
#    vs = np.reshape(vs,[-1,])
#    den = np.reshape(den,[-1,])
    
    n_ang = angle.shape[0]
    n_rc  = vp.shape[0]-1
    rc_zoep_pp = np.zeros((n_ang,len(vp)-1))



    for i in range(0, n_ang):
        for j in range(0, n_rc):
            rc_buf= rc_zoep(vp[j], vs[j], den[j], vp[j+1], vs[j+1], den[j+1], angle[i,0])        
            rc_zoep_pp[i,j] = rc_buf[0,0]
            
    return rc_zoep_pp