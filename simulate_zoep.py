import tensorflow as tf

def deg2rad(deg):
    
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180

def zoeppritz_rpp(vp1, vs1, rho1, vp2, vs2, rho2, theta1):
    
    theta1 = deg2rad(theta1)
    p      = tf.math.sin(theta1)/vp1
    temp = tf.math.minimum(tf.cast(1.0,tf.float32),tf.math.maximum(p*vp2,tf.cast(-1.0,tf.float32)))
    theta2 = tf.math.asin(temp);      # Transmission angle of P-wave
    temp = tf.math.minimum(tf.cast(1.0,tf.float32),tf.math.maximum(p*vs1,tf.cast(-1.0,tf.float32)))
    phi1   = tf.math.asin(temp);      # Reflection angle of converted S-wave
    temp = tf.math.minimum(tf.cast(1.0,tf.float32),tf.math.maximum(p*vs2,tf.cast(-1.0,tf.float32)))
    phi2   = tf.math.asin(temp);      # Transmission angle of converted S-wave

    a = rho2 * (1 - 2 * tf.math.sin(phi2)**2.) - rho1 * (1 - 2 * tf.math.sin(phi1)**2.)
    b = rho2 * (1 - 2 * tf.math.sin(phi2)**2.) + 2 * rho1 * tf.math.sin(phi1)**2.
    c = rho1 * (1 - 2 * tf.math.sin(phi1)**2.) + 2 * rho2 * tf.math.sin(phi2)**2.
    d = 2 * (rho2 * vs2**2 - rho1 * vs1**2)

    E = (b * tf.math.cos(theta1) / vp1) + (c * tf.math.cos(theta2) / vp2)
    F = (b * tf.math.cos(phi1) / vs1) + (c * tf.math.cos(phi2) / vs2)
    G = a - d * tf.math.cos(theta1)/vp1 * tf.math.cos(phi2)/vs2
    H = a - d * tf.math.cos(theta2)/vp2 * tf.math.cos(phi1)/vs1

    D = E*F + G*H*p**2

    rpp = (1/D) * (F*(b*(tf.math.cos(theta1)/vp1) - c*(tf.math.cos(theta2)/vp2)) \
                   - H*p**2 * (a + d*(tf.math.cos(theta1)/vp1)*(tf.math.cos(phi2)/vs2)))

    return tf.squeeze(rpp)

def simulate(vp, vs, den, angle):
    
    vp = tf.reshape(vp,[-1,])
    vs = tf.reshape(vs,[-1,])
    den = tf.reshape(den,[-1,]) 
    
    vp1, vp2 = vp[:-1], vp[1:]
    vs1, vs2 = vs[:-1], vs[1:]
    den1, den2 = den[:-1], den[1:]
    
    return zoeppritz_rpp(vp1, vs1, den1, vp2, vs2, den2, angle)  