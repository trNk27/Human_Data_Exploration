import os, sys
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.hgf.data import load_session
from analysis.hgf.model import make_session_models, SessionModel
from analysis.hgf.fit import _make_neg_log_post
from analysis.hgf.config import THETA_INIT, THETA_BOUNDS

sd = load_session("20250521")
m = SessionModel(sd)
vg = _make_neg_log_post([m], use_prior=True)
v,g = vg(jnp.asarray(THETA_INIT))
print("at init theta", THETA_INIT, "f=", float(v), "grad=", np.asarray(g))

def fg(theta):
    v,g = vg(jnp.asarray(theta, float))
    return float(v), np.asarray(g, float)

res = minimize(fg, THETA_INIT.copy(), jac=True, method="L-BFGS-B", bounds=THETA_BOUNDS,
               options={"maxiter":500,"ftol":1e-11,"gtol":1e-8})
print("LBFGS bounded: x=", res.x, "f=", res.fun, "nit=", res.nit, "ok=", res.success, "|", res.message)

res2 = minimize(fg, THETA_INIT.copy(), jac=True, method="L-BFGS-B",
               options={"maxiter":500,"ftol":1e-11,"gtol":1e-8})
print("LBFGS nobound: x=", res2.x, "f=", res2.fun, "nit=", res2.nit, "ok=", res2.success, "|", res2.message)
