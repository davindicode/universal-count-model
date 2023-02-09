bn = 40
rcov, neurons, tbin, resamples, rc_t, region_edge = HDC.get_dataset(session_id, phase, bn, '../scripts/data')

left_x = rcov[3].min()
right_x = rcov[3].max()
bottom_y = rcov[4].min()
top_y = rcov[4].max()

pick_neuron = list(range(neurons))


modes = [('GP', 'U', 'hd_w_s_pos_t', 64, 'identity', 3, [], False, 10, False, 'ew'), 
         ('GP', 'U', 'hd_w_s_pos_t_R1', 72, 'identity', 3, [6], False, 10, False, 'ew'), 
         ('GP', 'U', 'hd_w_s_pos_t_R2', 80, 'identity', 3, [6], False, 10, False, 'ew'), 
         ('GP', 'U', 'hd_w_s_pos_t_R3', 88, 'identity', 3, [6], False, 10, False, 'ew'), 
         ('GP', 'U', 'hd_w_s_pos_t_R4', 96, 'identity', 3, [6], False, 10, False, 'ew')]


### statistics over the behaviour ###
avg_models_z = []
var_models_z = []
ff_models_z = []

batch_size = 5000 # batching data to evaluate over all data
kcv = 2

bn = 40

for mode in modes:

    rcov, neurons, tbin, resamples, rc_t, region_edge = HDC.get_dataset(session_id, phase, bn, '../scripts/data')
    max_count = int(rc_t.max())
    x_counts = torch.arange(max_count+1)
    
    cvdata = model_utils.get_cv_sets(mode, [kcv], batch_size, rc_t, resamples, rcov)[0]
    full_model = get_full_model(session_id, phase, cvdata, resamples, bn, 
                                mode, rcov, max_count, neurons, gpu=gpu_dev)


    avg_model = []
    var_model = []
    ff_model = []

    for b in range(full_model.inputs.batches):
        P_mc = model_utils.compute_pred_P(full_model, b, pick_neuron, None, cov_samples=10, ll_samples=1, tr=0).cpu()

        avg = (x_counts[None, None, None, :]*P_mc).sum(-1)
        var = ((x_counts[None, None, None, :]**2*P_mc).sum(-1)-avg**2)
        ff = var/(avg+1e-12)
        avg_model.append(avg)
        var_model.append(var)
        ff_model.append(ff)

    avg_models_z.append(torch.cat(avg_model, dim=-1).mean(0).numpy())
    var_models_z.append(torch.cat(var_model, dim=-1).mean(0).numpy())
    ff_models_z.append(torch.cat(ff_model, dim=-1).mean(0).numpy())
    
    
b = 1
Pearson_ffz = []
ratioz = []

for d in range(len(avg_models_z)):
    Pearson_ffz_ = []
    ratioz_ = []
    for avg, ff in zip(avg_models_z[d], ff_models_z[d]):
        r, r_p = scstats.pearsonr(ff, avg) # Pearson r correlation test
        Pearson_ffz_.append((r, r_p))
        ratioz_.append(ff.std()/avg.std())
        
    Pearson_ffz.append(Pearson_ffz_)
    ratioz.append(ratioz_)
    
    
    
binning = 40
rcov, neurons, tbin, resamples, rc_t, region_edge = HDC.get_dataset(session_id, phase, binning, '../scripts/data')
max_count = int(rc_t.max())
x_counts = torch.arange(max_count+1)



# ELBO for models of different dimensions
kcvs = [2, 5, 8] # get corresponding training sets
Ms = modes[:5]
batch_size = 3000 # batching data to evaluate over all data

elbo = []
for em, mode in enumerate(Ms):
    for cvdata in model_utils.get_cv_sets(mode, kcvs, batch_size, rc_t, resamples, rcov):
        _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
        cv_set = (ftrain, fcov, vtrain, vcov)
        
        full_model = get_full_model(session_id, phase, cvdata, resamples, binning, 
                                    mode, rcov, max_count, neurons, gpu=gpu_dev)
        
        batches = full_model.likelihood.batches
        print(batches)
        elbo_ = []
        for b in range(batches):
            elbo_.append(full_model.objective(b, cov_samples=1, ll_mode='GH', bound='ELBO', neuron=None, 
                                              beta=1., ll_samples=100).data.cpu().numpy())
        elbo.append(np.array(elbo_).mean())
        
elbo = np.array(elbo).reshape(len(Ms), len(kcvs))



# cross validation for dimensionality
beta = 0.0
n_group = np.arange(5)
val_neuron = [n_group, n_group+5, n_group+10, n_group+15, n_group+20, n_group+25, np.arange(3)+30]
batch_size = 5000 # batching data to evaluate over all data

ncvx = 2
kcvs = [1, 2, 3, 5, 6, 8] # validation segments from splitting data into 10
Ms = modes[:5]

cv_pll = []
for em, mode in enumerate(Ms):
    for cvdata in model_utils.get_cv_sets(mode, kcvs, batch_size, rc_t, resamples, rcov):
        _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
        cv_set = (ftrain, fcov, vtrain, vcov)
        
        if em > 0:
            for v_neuron in val_neuron:
                fac = len(n_group)/len(v_neuron)
                
                prev_ll = np.inf
                for tr in range(ncvx):
                    full_model = get_full_model(session_id, phase, cvdata, resamples, binning, 
                                                mode, rcov, max_count, neurons, gpu=gpu_dev)
                    mask = np.ones((neurons,), dtype=bool)
                    mask[v_neuron] = False
                    f_neuron = np.arange(neurons)[mask]
                    ll = model_utils.LVM_pred_ll(full_model, mode[-5], mode[2], models.cov_used, cv_set, f_neuron, v_neuron, 
                                                 cov_MC=1, ll_MC=10, beta=beta, beta_z=0.0, max_iters=3000)[0]
                    if ll < prev_ll:
                        prev_ll = ll

                cv_pll.append(fac*prev_ll)
                
        else: # no latent
            for v_neuron in val_neuron:
                fac = len(n_group)/len(v_neuron)
                
                full_model = get_full_model(session_id, phase, cvdata, resamples, binning, 
                                            mode, rcov, max_count, neurons, gpu=gpu_dev)
                cv_pll.append(fac*model_utils.RG_pred_ll(full_model, mode[2], models.cov_used, cv_set, bound='ELBO', 
                                                         beta=beta, neuron_group=v_neuron, ll_mode='GH', ll_samples=100))

        
cv_pll = np.array(cv_pll).reshape(len(Ms), len(kcvs), len(val_neuron))



# get latent trajectories and drift timescale of neural tuning for 2D latent model
mode = modes[2]
batch_size = 5000


cvdata = model_utils.get_cv_sets(mode, [-1], batch_size, rc_t, resamples, rcov)[0]
full_model = get_full_model(session_id, phase, cvdata, resamples, binning, mode, rcov, max_count, 
                            neurons, gpu=gpu_dev)

X_loc, X_std = full_model.inputs.eval_XZ()

X_c = X_loc[6]
X_s = X_std[6]
z_tau = tbin/(1-torch.sigmoid(full_model.inputs.p_mu_6).data.cpu().numpy())

t_lengths = full_model.mapping.kernel.kern1.lengthscale[:, 0, 0, -3].data.cpu().numpy()



# load regression model with most input dimensions
mode = modes[4]
cvdata = model_utils.get_cv_sets(mode, [-1], 5000, rc_t, resamples, rcov)[0]
full_model = get_full_model(session_id, phase, cvdata, resamples, 40, mode, rcov, max_count, 
                            neurons, gpu=gpu_dev)



### head direction tuning ###
MC = 100

steps = 100
covariates = [np.linspace(0, 2*np.pi, steps), 
              0.*np.ones(steps), 0.*np.ones(steps), 
              (left_x+right_x)/2.*np.ones(steps), (bottom_y+top_y)/2.*np.ones(steps), 
              0.*np.ones(steps), 
              0.*np.ones(steps), 0.*np.ones(steps)]

P_mc = model_utils.compute_P(full_model, covariates, pick_neuron, MC=MC).cpu()


avg = (x_counts[None, None, None, :]*P_mc).sum(-1).mean(0).numpy()
pref_hd = covariates[0][np.argmax(avg, axis=1)]



# marginalized tuning curves
rcovz = list(rcov) + [X_c[:, 0], X_c[:, 1]]
MC = 10
skip = 10



### z ###
step = 100
P_tot = model_utils.marginalized_P(full_model, [np.linspace(-.2, .2, step)], [6], rcovz, 10000, 
                                   pick_neuron, MC=MC, skip=skip)
avg = (x_counts[None, None, None, :]*P_tot).sum(-1)
var = (x_counts[None, None, None, :]**2*P_tot).sum(-1)-avg**2
ff = var/avg

mz1_mean = avg.mean(0)
mz1_ff = ff.mean(0)
z1_mean_tf = (mz1_mean.max(dim=-1)[0] - mz1_mean.min(dim=-1)[0]) / (mz1_mean.max(dim=-1)[0] + mz1_mean.min(dim=-1)[0])
z1_ff_tf = (mz1_ff.max(dim=-1)[0] - mz1_ff.min(dim=-1)[0]) /(mz1_ff.max(dim=-1)[0] + mz1_ff.min(dim=-1)[0])



step = 100
P_tot = model_utils.marginalized_P(full_model, [np.linspace(-.2, .2, step)], [7], rcovz, 10000, 
                                   pick_neuron, MC=MC, skip=skip)
avg = (x_counts[None, None, None, :]*P_tot).sum(-1)
var = (x_counts[None, None, None, :]**2*P_tot).sum(-1)-avg**2
ff = var/avg

mz2_mean = avg.mean(0)
mz2_ff = ff.mean(0)
z2_mean_tf = (mz2_mean.max(dim=-1)[0] - mz2_mean.min(dim=-1)[0]) / (mz2_mean.max(dim=-1)[0] + mz2_mean.min(dim=-1)[0])
z2_ff_tf = (mz2_ff.max(dim=-1)[0] - mz2_ff.min(dim=-1)[0]) /(mz2_ff.max(dim=-1)[0] + mz2_ff.min(dim=-1)[0])




# compute 2D latent model properties of tuning curves and TI to latent space
z_d = 2

if z_d == 1: ### latent ###
    mean_z = []
    lower_z = []
    upper_z = []
    ffmean_z = []
    fflower_z = []
    ffupper_z = []

    steps = 100
    covariates_z = np.linspace(-.2, .2, steps)
    for en, n in enumerate(pick_neuron):
        # x_t, y_t, s_t, th_t, hd_t, time_t
        covariates = [pref_hd[n]*np.ones(steps), 0.*np.ones(steps), np.ones(steps)*0., 
                      (left_x+right_x)/2.*np.ones(steps), (bottom_y+top_y)/2.*np.ones(steps), 
                      0.*np.ones(steps), covariates_z]

        P_mc = model_utils.compute_P(full_model, covariates, [n], MC=1000).cpu()[:, 0, ...]

        avg = (x_counts[None, None, :]*P_mc).sum(-1)
        xcvar = ((x_counts[None, None, :]**2*P_mc).sum(-1)-avg**2)
        ff = xcvar/avg

        avgs = utils.signal.percentiles_from_samples(avg, percentiles=[0.05, 0.5, 0.95], 
                                                     smooth_length=5, padding_mode='replicate')
        lower, mean, upper = [cs_.cpu().numpy() for cs_ in avgs]

        ffs = utils.signal.percentiles_from_samples(ff, percentiles=[0.05, 0.5, 0.95], 
                                                smooth_length=5, padding_mode='replicate')
        fflower, ffmean, ffupper = [cs_.cpu().numpy() for cs_ in ffs]

        lower_z.append(lower)
        mean_z.append(mean)
        upper_z.append(upper)

        fflower_z.append(fflower)
        ffmean_z.append(ffmean)
        ffupper_z.append(ffupper)
    
else: ### 2d z ###
    grid_size_zz = (41, 41)
    grid_shape_zz = [[-.2, .2], [-.2, .2]]

    steps = np.product(grid_size_zz)
    A, B = grid_size_zz

    
    field_zz = []
    ff_zz = []
    t = 0
    for en, n in enumerate(pick_neuron):
        covariates = [pref_hd[n]*np.ones(steps), 
                      0.*np.ones(steps), 0.*np.ones(steps), 
                      (left_x+right_x)/2.*np.ones(steps), (bottom_y+top_y)/2.*np.ones(steps), 
                      t*np.ones(steps), 
                      np.linspace(-.2, .2, A)[:, None].repeat(B, axis=1).flatten(), 
                      np.linspace(-.2, .2, B)[None, :].repeat(A, axis=0).flatten()]

        P_mean = model_utils.compute_P(full_model, covariates, [n], MC=100).mean(0).cpu()
        avg = (x_counts[None, :]*P_mean[0, ...]).sum(-1).reshape(A, B).numpy()
        var = (x_counts[None, :]**2*P_mean[0, ...]).sum(-1).reshape(A, B).numpy()
        xcvar = (var-avg**2)

        field_zz.append(avg)
        ff_zz.append(xcvar/avg)

    field_zz = np.stack(field_zz)
    ff_zz = np.stack(ff_zz)
    
    
    
# KS framework for latent models, including Fisher Z scores
CV = [2, 5, 8]
bn = 40



### KS test ###
Qq = []
Zz = []
R = []
Rp = []

N = len(pick_neuron)
for kcv in CV:
    for en, mode in enumerate(modes):
        cvdata = model_utils.get_cv_sets(mode, [kcv], 3000, rc_t, resamples, rcov)[0]
        _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
        cv_set = (ftrain, fcov, vtrain, vcov)
        time_steps = ftrain.shape[-1]

        full_model = get_full_model(session_id, phase, cvdata, resamples, bn, 
                                    mode, rcov, max_count, neurons, gpu=gpu_dev)

        q_ = []
        Z_ = []
        for b in range(full_model.inputs.batches): # predictive posterior
            P_mc = model_utils.compute_pred_P(full_model, b, pick_neuron, None, cov_samples=10, ll_samples=1, tr=0)
            P = P_mc.mean(0).cpu().numpy()

            for n in range(N):
                spike_binned = full_model.likelihood.spikes[b][0, pick_neuron[n], :].numpy()
                q, Z = model_utils.get_q_Z(P[n, ...], spike_binned, deq_noise=None)
                q_.append(q)
                Z_.append(Z)

        q = []
        Z = []
        for n in range(N):
            q.append(np.concatenate(q_[n::N]))
            Z.append(np.concatenate(Z_[n::N]))


        Pearson_s = []
        for n in range(len(pick_neuron)):
            for m in range(n+1, len(pick_neuron)):
                r, r_p = scstats.pearsonr(Z[n], Z[m]) # Pearson r correlation test
                Pearson_s.append((r, r_p))

        r = np.array([p[0] for p in Pearson_s])
        r_p = np.array([p[1] for p in Pearson_s])

        Qq.append(q)
        Zz.append(Z)
        R.append(r)
        Rp.append(r_p)


fisher_z = []
fisher_q = []
for en, r in enumerate(R):
    fz = 0.5*np.log((1+r)/(1-r))*np.sqrt(time_steps-3)
    fisher_z.append(fz)
    fisher_q.append(utils.stats.Z_to_q(fz))

    
q_DS_ = []
T_DS_ = []
T_KS_ = []
for q in Qq:
    for qq in q:
        T_DS, T_KS, sign_DS, sign_KS, p_DS, p_KS = utils.stats.KS_statistics(qq, alpha=0.05, alpha_s=0.05)
        T_DS_.append(T_DS)
        T_KS_.append(T_KS)
        
        Z_DS = T_DS/np.sqrt(2/(qq.shape[0]-1))
        q_DS_.append(utils.stats.Z_to_q(Z_DS))
        
        
fisher_z = np.array(fisher_z).reshape(len(CV), len(Ms), -1)
fisher_q = np.array(fisher_q).reshape(len(CV), len(Ms), -1)

Qq = np.array(Qq).reshape(len(CV), len(Ms), len(pick_neuron), -1)
Zz = np.array(Zz).reshape(len(CV), len(Ms), len(pick_neuron), -1)
R = np.array(R).reshape(len(CV), len(Ms), len(pick_neuron), -1)
Rp = np.array(Rp).reshape(len(CV), len(Ms), len(pick_neuron), -1)
        
q_DS_ = np.array(q_DS_).reshape(len(CV), len(Ms), len(pick_neuron), -1)
T_DS_ = np.array(T_DS_).reshape(len(CV), len(Ms), len(pick_neuron), -1)
T_KS_ = np.array(T_KS_).reshape(len(CV), len(Ms), len(pick_neuron), -1)



T_KS_fishq = []
p_KS_fishq = []
for q in fisher_q:
    for qq in q:
        _, T_KS, _, _, _, p_KS = utils.stats.KS_statistics(qq, alpha=0.05, alpha_s=0.05)
        T_KS_fishq.append(T_KS)
        p_KS_fishq.append(p_KS)
        
T_KS_fishq = np.array(T_KS_fishq).reshape(len(CV), len(Ms))
p_KS_fishq = np.array(p_KS_fishq).reshape(len(CV), len(Ms))
        
        
T_KS_ks = []
p_KS_ks = []
for q in Qq:
    for qq in q:
        for qqq in qq:
            _, T_KS, _, _, _, p_KS = utils.stats.KS_statistics(qqq, alpha=0.05, alpha_s=0.05)
            T_KS_ks.append(T_KS)
            p_KS_ks.append(p_KS)
        
T_KS_ks = np.array(T_KS_ks).reshape(len(CV), len(Ms), len(pick_neuron))
p_KS_ks = np.array(p_KS_ks).reshape(len(CV), len(Ms), len(pick_neuron))



# delayed noise or spatiotemporal correlations
NN = len(pick_neuron)
delays = np.arange(5)
R_mat_spt = np.empty((len(Ms), len(delays), NN, NN))
R_mat_sptp = np.empty((len(Ms), len(delays), NN, NN))

kcv_ind = 1
for d, Z_ in enumerate(Zz[kcv_ind]):
    steps = len(Z_[0])-len(delays)
    
    for en, t in enumerate(delays):
        Pearson_s = []
        for n in range(NN):
            for m in range(NN):
                r, r_p = scstats.pearsonr(Z_[n][t:t+steps], Z_[m][:-len(delays)]) # Pearson r correlation test
                R_mat_spt[d, en, n, m] = r
                R_mat_sptp[d, en, n, m] = r_p

                
                
# compute timescales for input dimensions from ACG
delays = 5000
Tsteps = rcov[0].shape[0]
L = Tsteps-delays+1
acg_rc = []

for rc in rcov[:1]: # angular
    acg = np.empty(delays)
    for d in range(delays):
        A = rc[d:d+L]
        B = rc[:L]
        acg[d] = utils.stats.corr_circ_circ(A, B)
    acg_rc.append(acg)

for rc in rcov[1:-1]:
    acg = np.empty(delays)
    for d in range(delays):
        A = rc[d:d+L]
        B = rc[:L]
        acg[d] = ((A-A.mean())*(B-B.mean())).mean()/A.std()/B.std()
    acg_rc.append(acg)
    

acg_z = []
for rc in X_c.T:
    acg = np.empty(delays)
    for d in range(delays):
        A = rc[d:d+L]
        B = rc[:L]
        acg[d] = ((A-A.mean())*(B-B.mean())).mean()/A.std()/B.std()
    acg_z.append(acg)
    
    
timescales = []

for d in range(len(rcov)-1):
    timescales.append(np.where(acg_rc[d] < np.exp(-1))[0][0]*tbin)
    
for d in range(X_c.shape[-1]):
    timescales.append(np.where(acg_z[d] < np.exp(-1))[0][0]*tbin)

    
    
data_run = (
    avg_models_z, var_models_z, ff_models_z, 
    Pearson_ffz, ratioz, 
    X_c, X_s, cv_pll, elbo, z_tau, pref_hd, 
    grid_size_zz, grid_shape_zz, field_zz, ff_zz, 
    mz1_mean, mz1_ff, z1_mean_tf, z1_ff_tf, 
    mz2_mean, mz2_ff, z2_mean_tf, z2_ff_tf, 
    q_DS_, T_DS_, T_KS_, Qq, Zz, R, Rp, fisher_z, fisher_q, 
    T_KS_fishq, p_KS_fishq, T_KS_ks, p_KS_ks, 
    R_mat_spt, R_mat_sptp, 
    timescales, acg_rc, acg_z, t_lengths
)

pickle.dump(data_run, open('./saves/P_HDC_nc40.p', 'wb'))