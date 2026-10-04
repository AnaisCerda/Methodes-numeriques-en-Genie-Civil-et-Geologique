"""
advection.py : solveur d'advection 2D par volumes finis (forme conservative).

Conventions (les memes que dans le script des vitesses) :
    c[i, j]  : concentration moyenne de la cellule (i = ligne, j = colonne), taille (Ny, Nx)
    U[i, j]  : vitesse sur la face GAUCHE de la cellule (i, j), taille (Ny, Nx+1)
    V[i, j]  : vitesse sur la face HAUTE  de la cellule (i, j), taille (Ny+1, Nx)
    i augmente vers le bas de l'image, donc V > 0 veut dire "vers le bas".

Contenu :
    1. outils pour le labyrinthe (chargement, vitesses depuis phi, entree/sortie)
    2. pas de temps (CFL)
    3. flux aux faces : upwind, centre, MUSCL (ordre 2 avec limiteur)
    4. second membre rhs
    5. boucle en temps : simuler (Euler explicite ou RK2)
    6. cas tests analytiques : translation et rotation solide
"""
import time
import numpy as np


# ============================================================
# 1. Outils pour le labyrinthe
# ============================================================
def charger(dossier='.'):
    """Charge U.npy, V.npy et out_domain.txt."""
    U = np.load(f'{dossier}/U.npy')
    V = np.load(f'{dossier}/V.npy')
    dom = np.loadtxt(f'{dossier}/out_domain.txt', dtype=int)
    return U, V, dom


def preparer_labyrinthe(dom):
    """Renvoie le masque fluide, l'entree (i_in, j_in) et la sortie (i_out, j_out).
    Hypothese : l'entree est la ligne la plus haute contenant des 2, la sortie la plus basse."""
    fluid = dom > 0
    rows_bc = np.where((dom == 2).any(axis=1))[0]
    i_in, i_out = rows_bc.min(), rows_bc.max()
    j_in = np.where(dom[i_in, :] == 2)[0]
    j_out = np.where(dom[i_out, :] == 2)[0]
    return fluid, (i_in, j_in), (i_out, j_out)


def calcul_vitesses(phi, dom, dx=1.0, dy=1.0):
    """Meme calcul que dans le notebook 1 : phi (centres) -> U, V (faces), normalises (max = 1)."""
    Ny, Nx = phi.shape
    fluid = dom > 0
    U = np.zeros((Ny, Nx + 1))
    V = np.zeros((Ny + 1, Nx))
    face_U = fluid[:, 1:] & fluid[:, :-1]
    U[:, 1:Nx] = np.where(face_U, -(phi[:, 1:] - phi[:, :-1]) / dx, 0.0)
    face_V = fluid[1:, :] & fluid[:-1, :]
    V[1:Ny, :] = np.where(face_V, -(phi[1:, :] - phi[:-1, :]) / dy, 0.0)
    _, (i_in, j_in), (i_out, j_out) = preparer_labyrinthe(dom)
    V[i_in, j_in] = V[i_in + 1, j_in]
    V[i_out + 1, j_out] = V[i_out, j_out]
    vmax = max(np.abs(U).max(), np.abs(V).max())
    return U / vmax, V / vmax


# ============================================================
# 2. Pas de temps
# ============================================================
def pas_de_temps(U, V, dx, dy, cfl):
    """dt = cfl / max (somme des vitesses sortantes / taille de maille). Stable si cfl <= 1 (upwind)."""
    sortant = (np.maximum(U[:, 1:], 0) + np.maximum(-U[:, :-1], 0)) / dx \
            + (np.maximum(V[1:, :], 0) + np.maximum(-V[:-1, :], 0)) / dy
    return cfl / sortant.max()


# ============================================================
# 3. Flux aux faces
# ============================================================
def _fantomes(c, nb, inlet, c_in):
    """Entoure c de nb couches de cellules fantomes (valeur 0) et met c_in au-dessus de l'entree."""
    cp = np.pad(c, nb)
    if inlet is not None:
        i_in, j_in = inlet
        for k in range(nb):
            cp[i_in + nb - 1 - k, j_in + nb] = c_in     # fantome(s) situe(s) juste au-dessus de l'entree
    return cp


def flux_upwind(c, U, V, inlet=None, c_in=0.0):
    """c_face = valeur de la cellule d'ou vient le fluide (ordre 1)."""
    cp = _fantomes(c, 1, inlet, c_in)
    cL, cR = cp[1:-1, :-1], cp[1:-1, 1:]
    F = U * np.where(U > 0, cL, cR)
    cT, cB = cp[:-1, 1:-1], cp[1:, 1:-1]
    G = V * np.where(V > 0, cT, cB)
    return F, G


def flux_centre(c, U, V, inlet=None, c_in=0.0):
    """c_face = moyenne des deux cellules voisines (ordre 2, mais instable avec Euler explicite)."""
    cp = _fantomes(c, 1, inlet, c_in)
    F = U * 0.5 * (cp[1:-1, :-1] + cp[1:-1, 1:])
    G = V * 0.5 * (cp[:-1, 1:-1] + cp[1:, 1:-1])
    return F, G


def limiteur(a, b, kind):
    """Pente limitee a partir des differences a (gauche) et b (droite)."""
    if kind == 'minmod':
        return np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)
    if kind == 'vanleer':
        with np.errstate(divide='ignore', invalid='ignore'):
            r = 2 * a * b / (a + b)
        return np.where(a * b > 0, r, 0.0)
    if kind == 'fromm':                 # pas de limiteur : pente centree (oscille pres des fronts)
        return 0.5 * (a + b)
    if kind == 'aucun':                 # pente nulle = retombe sur l'upwind
        return np.zeros_like(a)
    raise ValueError("limiteur inconnu : " + kind)


def _flux_muscl_axe1(cp, fp, U, kind):
    """Flux MUSCL le long de l'axe 1. cp : c avec 2 fantomes de chaque cote (R, n+4),
    fp : masque fluide pareil, U : vitesses des faces (R, n+1)."""
    n = U.shape[1] - 1
    a = cp[:, 1:n + 3] - cp[:, 0:n + 2]
    b = cp[:, 2:n + 4] - cp[:, 1:n + 3]
    a = np.where(fp[:, 1:n + 3] & fp[:, 0:n + 2], a, 0.0)    # pas de pente a travers un mur
    b = np.where(fp[:, 2:n + 4] & fp[:, 1:n + 3], b, 0.0)
    s = limiteur(a, b, kind)                                  # pentes des cellules
    cL = cp[:, 1:n + 2] + 0.5 * s[:, 0:n + 1]                 # valeur a la face, cote gauche
    cR = cp[:, 2:n + 3] - 0.5 * s[:, 1:n + 2]                 # valeur a la face, cote droit
    return U * np.where(U > 0, cL, cR)


def flux_muscl(c, U, V, kind='minmod', inlet=None, c_in=0.0, fluid=None):
    """Reconstruction lineaire par morceaux (pente limitee) puis choix amont : ordre 2 en zone lisse."""
    cp = _fantomes(c, 2, inlet, c_in)
    if fluid is None:
        fp = np.ones(cp.shape, dtype=bool)
    else:
        fp = np.pad(fluid, 2)                                  # fantomes = murs (False)
        if inlet is not None:
            i_in, j_in = inlet
            fp[i_in, j_in + 2] = True
            fp[i_in + 1, j_in + 2] = True
    F = _flux_muscl_axe1(cp[2:-2, :], fp[2:-2, :], U, kind)
    G = _flux_muscl_axe1(cp[:, 2:-2].T, fp[:, 2:-2].T, V.T, kind).T
    return F, G


# ============================================================
# 4. Second membre
# ============================================================
def rhs(c, U, V, dx, dy, schema='upwind', lim='minmod', inlet=None, c_in=0.0, fluid=None):
    """dc/dt = -(F_droite - F_gauche)/dx - (G_bas - G_haut)/dy  (forme conservative)."""
    if schema == 'upwind':
        F, G = flux_upwind(c, U, V, inlet, c_in)
    elif schema == 'centre':
        F, G = flux_centre(c, U, V, inlet, c_in)
    elif schema == 'muscl':
        F, G = flux_muscl(c, U, V, lim, inlet, c_in, fluid)
    else:
        raise ValueError("schema inconnu : " + schema)
    dcdt = -(F[:, 1:] - F[:, :-1]) / dx - (G[1:, :] - G[:-1, :]) / dy
    if fluid is not None:
        dcdt = np.where(fluid, dcdt, 0.0)
    return dcdt, F, G


# ============================================================
# 5. Boucle en temps
# ============================================================
def simuler(c0, U, V, dx=1.0, dy=1.0, t_end=1.0, cfl=0.5,
            schema='upwind', lim='minmod', temps='euler',
            inlet=None, outlet=None, c_in=1.0, t_inj=None, fluid=None,
            snap_times=None, seuil_arrivee=None, arret_cmax=1e6, arret_sortie=None,
            progress=False):
    """
    Fait evoluer c de 0 a t_end.
      temps  : 'euler' (Euler explicite) ou 'rk2' (Heun, ordre 2 en temps)
      inlet  : (i_in, j_in) -> concentration c_in imposee au-dessus de l'entree (jusqu'a t_inj si donne)
      outlet : (i_out, j_out) -> enregistre la concentration de sortie
      snap_times    : liste d'instants ou on garde une copie de c
      seuil_arrivee : enregistre pour chaque cellule le 1er instant ou c >= seuil
      arret_cmax    : arrete le calcul si max|c| depasse cette valeur (instabilite)
      arret_sortie  : arrete le calcul quand la concentration de sortie atteint cette valeur
    Renvoie un dictionnaire de resultats.
    """
    c = c0.copy().astype(float)
    dt0 = pas_de_temps(U, V, dx, dy, cfl)
    n_total = int(np.ceil(t_end / dt0))
    freq = max(1, n_total // 10)
    aire = dx * dy
    masse = lambda cc: (cc[fluid].sum() if fluid is not None else cc.sum()) * aire

    t_list, m_list, cmax_l, cmin_l = [0.0], [masse(c)], [np.abs(c).max()], [c.min()]
    m_in = m_out = 0.0
    m_in_l, m_out_l, c_sortie = [0.0], [0.0], [np.nan]
    snap_times = [] if snap_times is None else sorted(snap_times)
    snaps, k = [], 0
    while k < len(snap_times) and snap_times[k] <= 1e-12:
        snaps.append((0.0, c.copy())); k += 1
    t_arr = np.full(c.shape, np.nan) if seuil_arrivee is not None else None

    t, n, explose = 0.0, 0, False
    t_start = time.perf_counter()
    while t < t_end - 1e-12:
        dt = min(dt0, t_end - t)
        cin = c_in if (t_inj is None or t < t_inj) else 0.0
        kw = dict(schema=schema, lim=lim, inlet=inlet, c_in=cin, fluid=fluid)

        d1, F1, G1 = rhs(c, U, V, dx, dy, **kw)
        if temps == 'euler':
            c = c + dt * d1
            F, G = F1, G1
        elif temps == 'rk2':
            d2, F2, G2 = rhs(c + dt * d1, U, V, dx, dy, **kw)
            c = c + 0.5 * dt * (d1 + d2)
            F, G = 0.5 * (F1 + F2), 0.5 * (G1 + G2)
        else:
            raise ValueError("temps inconnu : " + temps)
        t += dt
        n += 1

        # bilan de masse aux faces d'entree et de sortie (flux * largeur de la face)
        if inlet is not None:
            m_in += dt * G[inlet[0], inlet[1]].sum() * dx
        cs = np.nan
        if outlet is not None:
            i_out, j_out = outlet
            m_out += dt * G[i_out + 1, j_out].sum() * dx
            Vs = V[i_out + 1, j_out].sum()
            cs = G[i_out + 1, j_out].sum() / Vs if Vs > 0 else np.nan   # c moyenne ponderee par le debit

        t_list.append(t); m_list.append(masse(c))
        cmax_l.append(np.abs(c).max()); cmin_l.append(c.min())
        m_in_l.append(m_in); m_out_l.append(m_out); c_sortie.append(cs)

        if t_arr is not None:
            nouveau = (c >= seuil_arrivee) & np.isnan(t_arr)
            t_arr[nouveau] = t
        while k < len(snap_times) and t >= snap_times[k] - 1e-12:
            snaps.append((t, c.copy())); k += 1

        if progress and n % freq == 0:
            ecoule = time.perf_counter() - t_start
            frac = t / t_end
            print(f"  {100*frac:5.1f} %  | pas {n}/{n_total} | ecoule {ecoule:6.1f} s | restant ~{ecoule/frac - ecoule:6.1f} s")
        if not np.isfinite(cmax_l[-1]) or cmax_l[-1] > arret_cmax:
            explose = True
            break
        if arret_sortie is not None and np.isfinite(cs) and cs >= arret_sortie:
            break

    duree = time.perf_counter() - t_start
    return dict(c=c, dt=dt0, n=n, duree=duree, explose=explose, t_final=t,
                temps=np.array(t_list), masse=np.array(m_list),
                cmax=np.array(cmax_l), cmin=np.array(cmin_l),
                m_in=np.array(m_in_l), m_out=np.array(m_out_l),
                c_sortie=np.array(c_sortie), snaps=snaps, t_arrivee=t_arr)


def temps_arrivee(temps, c_sortie, seuil):
    """Premier instant ou la concentration de sortie atteint 'seuil' (nan si jamais)."""
    idx = np.where(np.nan_to_num(c_sortie, nan=-1.0) >= seuil)[0]
    return temps[idx[0]] if len(idx) else np.nan


def test_champ_constant(U, V, dom, schema='upwind', lim='minmod'):
    """Un champ c = 1 (entree comprise) doit rester constant : mesure l'effet de la divergence residuelle."""
    fluid, inlet, _ = preparer_labyrinthe(dom)
    c = np.where(fluid, 1.0, 0.0)
    d, _, _ = rhs(c, U, V, 1.0, 1.0, schema, lim, inlet, 1.0, fluid)
    return np.abs(d).max(), np.unravel_index(np.argmax(np.abs(d)), d.shape)


# ============================================================
# 6. Cas tests analytiques
# ============================================================
def ordre_observe(dxs, erreurs):
    """Pente de log(erreur) en fonction de log(dx) : ordre observe du schema."""
    return np.polyfit(np.log(dxs), np.log(erreurs), 1)[0]


def _erreurs(c, c_ex, dx, dy):
    e = c - c_ex
    return dict(L1=np.abs(e).sum() * dx * dy,
                L2=np.sqrt((e ** 2).sum() * dx * dy),
                Linf=np.abs(e).max())


def cas_translation(N=100, profil='gauss', schema='upwind', lim='minmod', temps='euler',
                    cfl=0.5, t_end=30.0, u0=1.0, Lx=100.0, Ly=50.0, x0=25.0, sigma=5.0, **kw):
    """Une tache translatee a la vitesse u0 (vers la droite) dans un domaine Lx x Ly.
    N = nombre de cellules en x (Ny = N * Ly / Lx). profil : 'gauss' ou 'creneau'."""
    Nx = N
    Ny = int(round(N * Ly / Lx))
    dx = Lx / Nx
    dy = dx
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y)
    y0 = Ly / 2

    def forme(xc):
        if profil == 'gauss':
            return np.exp(-((X - xc) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
        if profil == 'creneau':
            return ((np.abs(X - xc) <= 1.5 * sigma) & (np.abs(Y - y0) <= 1.5 * sigma)).astype(float)
        raise ValueError("profil inconnu : " + profil)

    c0 = forme(x0)
    U = u0 * np.ones((Ny, Nx + 1))
    V = np.zeros((Ny + 1, Nx))
    res = simuler(c0, U, V, dx, dy, t_end, cfl, schema, lim, temps, **kw)
    c_ex = forme(x0 + u0 * res['t_final'])
    res.update(c0=c0, c_exact=c_ex, x=x, y=y, dx=dx, dy=dy, N=N,
               err=_erreurs(res['c'], c_ex, dx, dy))
    return res


def cas_rotation(N=100, schema='upwind', lim='minmod', temps='euler', cfl=0.5,
                 omega=0.02, tours=1.0, L=100.0, sigma=4.0, R=15.0, **kw):
    """Rotation solide u = -omega (y - yc), v = omega (x - xc) dans un carre L x L.
    Une gaussienne a la distance R du centre fait 'tours' tours et doit revenir a sa place."""
    dx = dy = L / N
    x = (np.arange(N) + 0.5) * dx
    y = (np.arange(N) + 0.5) * dy
    X, Y = np.meshgrid(x, y)
    xc = yc = L / 2
    xf = np.arange(N + 1) * dx                      # positions des faces
    yf = np.arange(N + 1) * dy
    U = -omega * (y[:, None] - yc) * np.ones((N, N + 1))      # U[i, j] : face x_j, hauteur y_i
    V = omega * (x[None, :] - xc) * np.ones((N + 1, N))       # V[i, j] : face y_i, abscisse x_j

    def forme(theta):
        xg, yg = xc + R * np.cos(theta), yc + R * np.sin(theta)
        return np.exp(-((X - xg) ** 2 + (Y - yg) ** 2) / (2 * sigma ** 2))

    theta0 = np.pi / 2
    c0 = forme(theta0)
    t_end = tours * 2 * np.pi / omega
    res = simuler(c0, U, V, dx, dy, t_end, cfl, schema, lim, temps, **kw)
    c_ex = forme(theta0 + omega * res['t_final'])
    res.update(c0=c0, c_exact=c_ex, x=x, y=y, dx=dx, dy=dy, N=N,
               err=_erreurs(res['c'], c_ex, dx, dy))
    return res
