session_id = 'Mouse28-140313'
phase = 'wake'


### regression ###
modes = [('GP', 'U', 'hd', 8, 'identity', 3, [], False, 10, False, 'ew'), 
         ('GP', 'U', 'hd_w_s_t', 48, 'identity', 3, [], False, 10, False, 'ew'), 
         ('GP', 'U', 'hd_w_s_pos_t', 64, 'identity', 3, [], False, 10, False, 'ew'), 
         ('GP', 'IP', 'hd_w_s_pos_t', 64, 'exp', 1, [], False, 10, False, 'ew'), 
         ('GP', 'hNB', 'hd_w_s_pos_t', 64, 'exp', 1, [], False, 10, False, 'ew'), 
         ('GP', 'U', 'hd_w_s_pos_t', 64, 'identity', 3, [], False, 10, False, 'qd')]

bn = 40
rcov, neurons, tbin, resamples, rc_t, region_edge = HDC.get_dataset(session_id, phase, bn, '../scripts/data')

left_x = rcov[3].min()
right_x = rcov[3].max()
bottom_y = rcov[4].min()
top_y = rcov[4].max()

pick_neuron = list(range(neurons))



### statistics over the behaviour ###
avg_models = []
var_models = []
ff_models = []

binnings = [20, 40, 100, 200, 500]

for bn in binnings:

    rcov, neurons, tbin, resamples, rc_t, region_edge = HDC.get_dataset(session_id, phase, bn, '../scripts/data')
    max_count = int(rc_t.max())
    x_counts = torch.arange(max_count+1)
    
    mode = modes[2]
    cvdata = model_utils.get_cv_sets(mode, [2], 5000, rc_t, resamples, rcov)[0]
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

    avg_models.append(torch.cat(avg_model, dim=-1).mean(0).numpy())
    var_models.append(torch.cat(var_model, dim=-1).mean(0).numpy())
    ff_models.append(torch.cat(ff_model, dim=-1).mean(0).numpy())
    
    
    
    
# KS framework for regression models
CV = [2, 5, 8]

### KS test over binnings ###
Qq_bn = []
Zz_bn = []
R_bn = []
Rp_bn = []
mode = modes[2]

N = len(pick_neuron)
for kcv in CV:
    for en, bn in enumerate(binnings):
        cvdata = model_utils.get_cv_sets(mode, [kcv], 3000, rc_t, resamples, rcov)[0]
        _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
        cv_set = (ftrain, fcov, vtrain, vcov)
        time_steps = ftrain.shape[-1]

        full_model = get_full_model(session_id, phase, cvdata, resamples, bn, 
                                    mode, rcov, max_count, neurons, gpu=gpu_dev)

        if en == 0:
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

        elif en > 0:
            cov_used = models.cov_used(mode[2], fcov)
            q = model_utils.compute_count_stats(full_model, mode[1], tbin, ftrain, cov_used, pick_neuron, \
                                           traj_len=1, start=0, T=time_steps, bs=5000)
            Z = [utils.stats.q_to_Z(q_) for q_ in q]    

        Pearson_s = []
        for n in range(len(pick_neuron)):
            for m in range(n+1, len(pick_neuron)):
                r, r_p = scstats.pearsonr(Z[n], Z[m]) # Pearson r correlation test
                Pearson_s.append((r, r_p))

        r = np.array([p[0] for p in Pearson_s])
        r_p = np.array([p[1] for p in Pearson_s])

        Qq_bn.append(q)
        Zz_bn.append(Z)
        R_bn.append(r)
        Rp_bn.append(r_p)
        
        
q_DS_bn = []
T_DS_bn = []
T_KS_bn = []
for q in Qq_bn:
    for qq in q:
        T_DS, T_KS, sign_DS, sign_KS, p_DS, p_KS = utils.stats.KS_statistics(qq, alpha=0.05, alpha_s=0.05)
        T_DS_ll.append(T_DS)
        T_KS_ll.append(T_KS)
        
        Z_DS = T_DS/np.sqrt(2/(qq.shape[0]-1))
        q_DS_ll.append(utils.stats.Z_to_q(Z_DS))

Qq_bn = np.array(Qq_bn).reshape(len(CV), len(binnings), -1)
Zz_bn = np.array(Zz_bn).reshape(len(CV), len(binnings), -1)
R_bn = np.array(R_bn).reshape(len(CV), len(binnings), -1)
Rp_bn = np.array(Rp_bn).reshape(len(CV), len(binnings), -1)
        
q_DS_bn = np.array(q_DS_bn).reshape(len(CV), len(binnings), -1)
T_DS_bn = np.array(T_DS_bn).reshape(len(CV), len(binnings), -1)
T_KS_bn = np.array(T_KS_bn).reshape(len(CV), len(binnings), -1)



bn = 40
rcov, neurons, tbin, resamples, rc_t, region_edge = HDC.get_dataset(session_id, phase, bn, '../scripts/data')
max_count = int(rc_t.max())
x_counts = torch.arange(max_count+1)

HD_offset = -1.0 # global shift of head direction coordinates, makes plots better as the preferred head directions are not at axis lines



# cross validation
PLL_rg_ll = []
PLL_rg_cov = []
kcvs = [1, 2, 3, 5, 6, 8] # validation segments from splitting data into 10

beta = 0.0
batchsize = 5000

PLL_rg_ll = []
Ms = modes[2:5]
for mode in Ms: # likelihood
    
    for cvdata in model_utils.get_cv_sets(mode, kcvs, batchsize, rc_t, resamples, rcov):
        _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
        cv_set = (ftrain, fcov, vtrain, vcov)
    
        full_model = get_full_model(session_id, phase, cvdata, resamples, bn, mode, 
                                    rcov, max_count, neurons, gpu=gpu_dev)
        PLL_rg_ll.append(model_utils.RG_pred_ll(full_model, mode[2], models.cov_used, cv_set, bound='ELBO', 
                                           beta=beta, neuron_group=None, ll_mode='GH', ll_samples=100))
    
PLL_rg_ll = np.array(PLL_rg_ll).reshape(len(Ms), len(kcvs))



CV = [2, 5, 8] # validation segments from splitting data into 10

### KS test ###
Qq_ll = []
Zz_ll = []
R_ll = []
Rp_ll = []

batch_size = 3000
N = len(pick_neuron)
for kcv in CV:
    for en, mode in enumerate(Ms):
        cvdata = model_utils.get_cv_sets(mode, [kcv], batch_size, rc_t, resamples, rcov)[0]
        _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
        cv_set = (ftrain, fcov, vtrain, vcov)
        time_steps = ftrain.shape[-1]

        full_model = get_full_model(session_id, phase, cvdata, resamples, bn, 
                                    mode, rcov, max_count, neurons, gpu=gpu_dev)

        if en == 0:
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

        elif en > 0:
            cov_used = models.cov_used(mode[2], fcov)
            q = model_utils.compute_count_stats(full_model, mode[1], tbin, ftrain, cov_used, pick_neuron, \
                                            traj_len=1, start=0, T=time_steps, bs=5000)
            Z = [utils.stats.q_to_Z(q_) for q_ in q]    

        Pearson_s = []
        for n in range(len(pick_neuron)):
            for m in range(n+1, len(pick_neuron)):
                r, r_p = scstats.pearsonr(Z[n], Z[m]) # Pearson r correlation test
                Pearson_s.append((r, r_p))

        r = np.array([p[0] for p in Pearson_s])
        r_p = np.array([p[1] for p in Pearson_s])

        Qq_ll.append(q)
        Zz_ll.append(Z)
        R_ll.append(r)
        Rp_ll.append(r_p)
        
        
q_DS_ll = []
T_DS_ll = []
T_KS_ll = []
for q in Qq_ll:
    for qq in q:
        T_DS, T_KS, sign_DS, sign_KS, p_DS, p_KS = utils.stats.KS_statistics(qq, alpha=0.05, alpha_s=0.05)
        T_DS_ll.append(T_DS)
        T_KS_ll.append(T_KS)
        
        Z_DS = T_DS/np.sqrt(2/(qq.shape[0]-1))
        q_DS_ll.append(utils.stats.Z_to_q(Z_DS))

Qq_ll = np.array(Qq_ll).reshape(len(CV), len(Ms), -1)
Zz_ll = np.array(Zz_ll).reshape(len(CV), len(Ms), -1)
R_ll = np.array(R_ll).reshape(len(CV), len(Ms), -1)
Rp_ll = np.array(Rp_ll).reshape(len(CV), len(Ms), -1)
        
q_DS_ll = np.array(q_DS_ll).reshape(len(CV), len(Ms), -1)
T_DS_ll = np.array(T_DS_ll).reshape(len(CV), len(Ms), -1)
T_KS_ll = np.array(T_KS_ll).reshape(len(CV), len(Ms), -1)



PLL_rg_cov = []
kcvs = [1, 2, 3, 5, 6, 8] # validation segments from splitting data into 10

Ms = modes[:3]
for mode in Ms: # input space
    
    for cvdata in model_utils.get_cv_sets(mode, kcvs, batchsize, rc_t, resamples, rcov):
        _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
        cv_set = (ftrain, fcov, vtrain, vcov)
    
        full_model = get_full_model(session_id, phase, cvdata, resamples, bn, mode, 
                                    rcov, max_count, neurons, gpu=gpu_dev)
        PLL_rg_cov.append(model_utils.RG_pred_ll(full_model, mode[2], models.cov_used, cv_set, bound='ELBO', 
                                                 beta=beta, neuron_group=None, ll_mode='GH', ll_samples=100))
    
PLL_rg_cov = np.array(PLL_rg_cov).reshape(len(Ms), len(kcvs))



# load universal regression model
mode = modes[2]
kcv = -1 # fit on the full dataset
cvdata = model_utils.get_cv_sets(mode, [kcv], 3000, rc_t, resamples, rcov)[0]
full_model = get_full_model(session_id, phase, cvdata, resamples, bn, mode, rcov, 
                            max_count, neurons, gpu=gpu_dev)

TT = tbin*resamples



# marginalized tuning curves
MC = 100
skip = 10
batch_size = 10000


### hd ###
steps = 100
P_tot = model_utils.marginalized_P(full_model, [np.linspace(0, 2*np.pi, steps)], [0], rcov, batch_size, 
                              pick_neuron, MC=MC, skip=skip)
avg = (x_counts[None, None, None, :]*P_tot).sum(-1)
var = (x_counts[None, None, None, :]**2*P_tot).sum(-1)-avg**2
ff = var/avg

avgs = utils.signal.percentiles_from_samples(avg, percentiles=[0.05, 0.5, 0.95], 
                                             smooth_length=5, padding_mode='circular')
mhd_lower, mhd_mean, mhd_upper = [cs_.cpu().numpy() for cs_ in avgs]

ffs = utils.signal.percentiles_from_samples(ff, percentiles=[0.05, 0.5, 0.95], 
                                            smooth_length=5, padding_mode='circular')
mhd_fflower, mhd_ffmean, mhd_ffupper = [cs_.cpu().numpy() for cs_ in ffs]

# total variance decomposition
hd_mean_EV = avg.var(0).mean(-1)
hd_mean_VE = avg.mean(0).var(-1)
hd_ff_EV = avg.var(0).mean(-1)
hd_ff_VE = avg.mean(0).var(-1)

# TI
hd_mean_tf = (mhd_mean.max(dim=-1)[0] - mhd_mean.min(dim=-1)[0]) / (mhd_mean.max(dim=-1)[0] + mhd_mean.min(dim=-1)[0])
hd_ff_tf = (mhd_ffmean.max(dim=-1)[0] - mhd_ffmean.min(dim=-1)[0]) /(mhd_ffmean.max(dim=-1)[0] + mhd_ffmean.min(dim=-1)[0])


### omega ###
steps = 100
w_edge = (-rcov[1].min()+rcov[1].max())/2.
covariates_w = np.linspace(-w_edge, w_edge, steps)
P_tot = model_utils.marginalized_P(full_model, [covariates_w], [1], rcov, batch_size, 
                                   pick_neuron, MC=MC, skip=skip)
avg = (x_counts[None, None, None, :]*P_tot).sum(-1)
var = (x_counts[None, None, None, :]**2*P_tot).sum(-1)-avg**2
ff = var/avg

mw_mean = avg.mean(0)
mw_ff = ff.mean(0)
w_mean_tf = (mw_mean.max(dim=-1)[0] - mw_mean.min(dim=-1)[0]) / (mw_mean.max(dim=-1)[0] + mw_mean.min(dim=-1)[0])
w_ff_tf = (mw_ff.max(dim=-1)[0] - mw_ff.min(dim=-1)[0]) /(mw_ff.max(dim=-1)[0] + mw_ff.min(dim=-1)[0])


### speed ###
steps = 100
P_tot = model_utils.marginalized_P(full_model, [np.linspace(0, 30., steps)], [2], rcov, batch_size, 
                                   pick_neuron, MC=MC, skip=skip)
avg = (x_counts[None, None, None, :]*P_tot).sum(-1)
var = (x_counts[None, None, None, :]**2*P_tot).sum(-1)-avg**2
ff = var/avg

ms_mean = avg.mean(0)
ms_ff = ff.mean(0)
s_mean_tf = (ms_mean.max(dim=-1)[0] - ms_mean.min(dim=-1)[0]) / (ms_ff.max(dim=-1)[0] + ms_ff.min(dim=-1)[0])
s_ff_tf = (ms_ff.max(dim=-1)[0] - ms_ff.min(dim=-1)[0]) /(ms_ff.max(dim=-1)[0] + ms_ff.min(dim=-1)[0])


### time ###
steps = 100
P_tot = model_utils.marginalized_P(full_model, [np.linspace(0, TT, steps)], [5], rcov, batch_size, 
                                   pick_neuron, MC=MC, skip=skip)
avg = (x_counts[None, None, None, :]*P_tot).sum(-1)
var = (x_counts[None, None, None, :]**2*P_tot).sum(-1)-avg**2
ff = var/avg

mt_mean = avg.mean(0)
mt_ff = ff.mean(0)
t_mean_tf = (mt_mean.max(dim=-1)[0] - mt_mean.min(dim=-1)[0]) / (mt_ff.max(dim=-1)[0] + mt_ff.min(dim=-1)[0])
t_ff_tf = (mt_ff.max(dim=-1)[0] - mt_ff.min(dim=-1)[0]) /(mt_ff.max(dim=-1)[0] + mt_ff.min(dim=-1)[0])



### position ###
grid_size_pos = (12, 10)
grid_shape_pos = [[left_x, right_x], [bottom_y, top_y]]

steps = np.product(grid_size_pos)
A, B = grid_size_pos

cov_list = [np.linspace(left_x, right_x, A)[:, None].repeat(B, axis=1).flatten(), 
            np.linspace(bottom_y, top_y, B)[None, :].repeat(A, axis=0).flatten()]
                      
P_tot = model_utils.marginalized_P(full_model, cov_list, [3, 4], rcov, 10000, 
                                   pick_neuron, MC=MC, skip=skip)
avg = (x_counts[None, None, None, :]*P_tot).sum(-1)
var = (x_counts[None, None, None, :]**2*P_tot).sum(-1)-avg**2
ff = var/avg

mpos_mean = avg.mean(0)
mpos_ff = ff.mean(0)
pos_mean_tf = (mpos_mean.max(dim=-1)[0] - mpos_mean.min(dim=-1)[0]) / (mpos_mean.max(dim=-1)[0] + mpos_mean.min(dim=-1)[0])
pos_ff_tf = (mpos_ff.max(dim=-1)[0] - mpos_ff.min(dim=-1)[0]) / (mpos_ff.max(dim=-1)[0] + mpos_ff.min(dim=-1)[0])
mpos_mean = mpos_mean.reshape(-1, A, B)
mpos_ff = mpos_ff.reshape(-1, A, B)




# conditional tuning curves
MC = 300
MC_ = 100


### head direction tuning ###
steps = 100
covariates = [np.linspace(0, 2*np.pi, steps)-HD_offset, 
              0.*np.ones(steps), 0.*np.ones(steps), 
              (left_x+right_x)/2.*np.ones(steps), (bottom_y+top_y)/2.*np.ones(steps), 
              0.*np.ones(steps)]

P_mc = model_utils.compute_P(full_model, covariates, pick_neuron, MC=MC).cpu()


avg = (x_counts[None, None, None, :]*P_mc).sum(-1)
xcvar = ((x_counts[None, None, None, :]**2*P_mc).sum(-1)-avg**2)
ff = xcvar/avg

avgs = utils.signal.percentiles_from_samples(avg, percentiles=[0.05, 0.5, 0.95], 
                                             smooth_length=5, padding_mode='circular')
lower_hd, mean_hd, upper_hd = [cs_.cpu().numpy() for cs_ in avgs]

ffs = utils.signal.percentiles_from_samples(ff, percentiles=[0.05, 0.5, 0.95], 
                                            smooth_length=5, padding_mode='circular')
fflower_hd, ffmean_hd, ffupper_hd = [cs_.cpu().numpy() for cs_ in ffs]

covariates_hd = np.linspace(0, 2*np.pi, steps)



### hd_w ###
grid_size_hdw = (51, 41)
grid_shape_hdw = [[0, 2*np.pi], [-10., 10.]]

steps = np.product(grid_size_hdw)
A, B = grid_size_hdw
covariates = [np.linspace(0, 2*np.pi, A)[:, None].repeat(B, axis=1).flatten(), 
              np.linspace(-10., 10., B)[None, :].repeat(A, axis=0).flatten(), 
              0.*np.ones(steps), 
              (left_x+right_x)/2.*np.ones(steps), (bottom_y+top_y)/2.*np.ones(steps), 
              0.*np.ones(steps)]

P_mean = model_utils.compute_P(full_model, covariates, pick_neuron, MC=MC).mean(0).cpu()
field_hdw = (x_counts[None, None, :]*P_mean).sum(-1).reshape(-1, A, B).numpy()



# compute preferred HD
grid = (101, 21)
grid_shape = [[0, 2*np.pi], [-10., 10.]]

steps = np.product(grid)
A, B = grid

w_arr = np.linspace(-10., 10., B)
covariates = [np.linspace(0, 2*np.pi, A)[:, None].repeat(B, axis=1).flatten(), 
              w_arr[None, :].repeat(A, axis=0).flatten(), 
              0.*np.ones(steps), 
              (left_x+right_x)/2.*np.ones(steps), (bottom_y+top_y)/2.*np.ones(steps), 
              0.*np.ones(steps)]

P_mean = model_utils.compute_P(full_model, covariates, pick_neuron, MC=MC).mean(0).cpu()
field = (x_counts[None, None, :]*P_mean).sum(-1).reshape(-1, A, B).numpy()



Z = np.cos(covariates[0]) + np.sin(covariates[0])*1j # CoM angle
Z = Z[None, :].reshape(-1, A, B)
pref_hdw = (np.angle((Z*field).mean(1)) % (2*np.pi)) # neurons, w


# ATI
ATI = []
res_var = []
for k in range(neurons):
    _, a, shift, losses = utils.signal.circ_lin_regression(pref_hdw[k, :], w_arr/(2*np.pi), dev='cpu', iters=1000, lr=1e-2)
    ATI.append(-a)
    res_var.append(losses[-1])
ATI = np.array(ATI)
res_var = np.array(res_var)




### omega tuning ###
mean_w = []
lower_w = []
upper_w = []
ffmean_w = []
fflower_w = []
ffupper_w = []

steps = 100
w_edge = (-rcov[1].min()+rcov[1].max())/2.
covariates_w = np.linspace(-w_edge, w_edge, steps)
for en, n in enumerate(pick_neuron):
    covariates = [pref_hdw[en, len(w_arr)//2]*np.ones(steps), 
                  covariates_w, 
                  0.*np.ones(steps), 
                  (left_x+right_x)/2.*np.ones(steps), (bottom_y+top_y)/2.*np.ones(steps), 
                  0.*np.ones(steps)]

    P_mc = model_utils.compute_P(full_model, covariates, [n], MC=MC)[:, 0, ...].cpu()

    avg = (x_counts[None, None, :]*P_mc).sum(-1)
    xcvar = ((x_counts[None, None, :]**2*P_mc).sum(-1)-avg**2)
    ff = xcvar/avg

    avgs = utils.signal.percentiles_from_samples(avg, percentiles=[0.05, 0.5, 0.95], 
                                                 smooth_length=5, padding_mode='replicate')
    lower, mean, upper = [cs_.cpu().numpy() for cs_ in avgs]

    ffs = utils.signal.percentiles_from_samples(ff, percentiles=[0.05, 0.5, 0.95], 
                                                smooth_length=5, padding_mode='replicate')
    fflower, ffmean, ffupper = [cs_.cpu().numpy() for cs_ in ffs]
    
    lower_w.append(lower)
    mean_w.append(mean)
    upper_w.append(upper)
    
    fflower_w.append(fflower)
    ffmean_w.append(ffmean)
    ffupper_w.append(ffupper)




### hd_t ###
grid_size_hdt = (51, 41)
grid_shape_hdt = [[0, 2*np.pi], [0., TT]]

steps = np.product(grid_size_hdt)
A, B = grid_size_hdt
covariates = [np.linspace(0, 2*np.pi, A)[:, None].repeat(B, axis=1).flatten(), 
              0.*np.ones(steps), 0.*np.ones(steps), 
              (left_x+right_x)/2.*np.ones(steps), (bottom_y+top_y)/2.*np.ones(steps), 
              np.linspace(0., TT, B)[None, :].repeat(A, axis=0).flatten()]

P_mean = model_utils.compute_P(full_model, covariates, pick_neuron, MC=MC_).mean(0).cpu()
field_hdt = (x_counts[None, None, :]*P_mean).sum(-1).reshape(-1, A, B).numpy()



# drift and similarity matrix
grid = (201, 16)
grid_shape = [[0, 2*np.pi], [0., TT]]

steps = np.product(grid)
A, B = grid

t_arr = np.linspace(0., TT, B)
dt_arr = t_arr[1]-t_arr[0]
covariates = [np.linspace(0, 2*np.pi, A)[:, None].repeat(B, axis=1).flatten(), 
              0.*np.ones(steps), 0.*np.ones(steps), 
              (left_x+right_x)/2.*np.ones(steps), (bottom_y+top_y)/2.*np.ones(steps), 
              t_arr[None, :].repeat(A, axis=0).flatten()]

P_mean = model_utils.compute_P(full_model, covariates, pick_neuron, MC=MC_).mean(0).cpu()
field = (x_counts[None, None, :]*P_mean).sum(-1).reshape(-1, A, B).numpy()



Z = np.cos(covariates[0]) + np.sin(covariates[0])*1j # CoM angle
Z = Z[None, :].reshape(-1, A, B)
E_exp = (Z*field).sum(-2)/field.sum(-2)
pref_hdt = (np.angle(E_exp) % (2*np.pi)) # neurons, t

tun_width = 1.-np.abs(E_exp)
amp_t = field.mean(-2) # mean amplitude
ampm_t = field.max(-2)

sim_mat = []
act = (field-field.mean(-2, keepdims=True))/field.std(-2, keepdims=True)
en = np.argsort(pref_hdt, axis=0)
for t in range(B):
    a = act[en[:, t], :, t]
    sim_mat = ((a[:, None, :]*a[None, ...]).mean(-1))



drift = []
res_var_drift = []
for k in range(len(pick_neuron)):
    _, a, shift, losses = utils.signal.circ_lin_regression(pref_hdt[k, :], t_arr/(2*np.pi)/1e2, 
                                                           dev='cpu', iters=1000, lr=1e-2)
    drift.append(a/1e2)
    res_var_drift.append(losses[-1])
drift = np.array(drift)
res_var_drift = np.array(res_var_drift)




### speed ###
mean_s = []
lower_s = []
upper_s = []
ffmean_s = []
fflower_s = []
ffupper_s = []
    
steps = 100
covariates_s = np.linspace(0, 30., steps)
for en, n in enumerate(pick_neuron):
    covariates = [pref_hdw[en, len(w_arr)//2]*np.ones(steps), 
                  0.*np.ones(steps), covariates_s, 
                  (left_x+right_x)/2.*np.ones(steps), (bottom_y+top_y)/2.*np.ones(steps), 
                  0.*np.ones(steps)]

    P_mc = model_utils.compute_P(full_model, covariates, [n], MC=MC)[:, 0, ...].cpu()

    avg = (x_counts[None, None, :]*P_mc).sum(-1)
    xcvar = ((x_counts[None, None, :]**2*P_mc).sum(-1)-avg**2)
    ff = xcvar/avg

    avgs = utils.signal.percentiles_from_samples(avg, percentiles=[0.05, 0.5, 0.95], 
                                                 smooth_length=5, padding_mode='replicate')
    lower, mean, upper = [cs_.cpu().numpy() for cs_ in avgs]
    
    ffs = utils.signal.percentiles_from_samples(ff, percentiles=[0.05, 0.5, 0.95], 
                                                smooth_length=5, padding_mode='replicate')
    fflower, ffmean, ffupper = [cs_.cpu().numpy() for cs_ in ffs]

    lower_s.append(lower)
    mean_s.append(mean)
    upper_s.append(upper)
    
    fflower_s.append(fflower)
    ffmean_s.append(ffmean)
    ffupper_s.append(ffupper)
    
    
    
    
### time ###
mean_t = []
lower_t = []
upper_t = []
ffmean_t = []
fflower_t = []
ffupper_t = []
    
steps = 100
covariates_t = np.linspace(0, TT, steps)
for en, n in enumerate(pick_neuron):
    covariates = [pref_hdw[en, len(w_arr)//2]*np.ones(steps), 
                  0.*np.ones(steps), 0.*np.ones(steps), 
                  (left_x+right_x)/2.*np.ones(steps), (bottom_y+top_y)/2.*np.ones(steps), 
                  covariates_t]

    P_mc = model_utils.compute_P(full_model, covariates, [n], MC=MC)[:, 0, ...].cpu()

    avg = (x_counts[None, None, :]*P_mc).sum(-1)
    xcvar = ((x_counts[None, None, :]**2*P_mc).sum(-1)-avg**2)
    ff = xcvar/avg

    avgs = utils.signal.percentiles_from_samples(avg, percentiles=[0.05, 0.5, 0.95], 
                                                 smooth_length=5, padding_mode='replicate')
    lower, mean, upper = [cs_.cpu().numpy() for cs_ in avgs]
    
    ffs = utils.signal.percentiles_from_samples(ff, percentiles=[0.05, 0.5, 0.95], 
                                                smooth_length=5, padding_mode='replicate')
    fflower, ffmean, ffupper = [cs_.cpu().numpy() for cs_ in ffs]

    lower_t.append(lower)
    mean_t.append(mean)
    upper_t.append(upper)
    
    fflower_t.append(fflower)
    ffmean_t.append(ffmean)
    ffupper_t.append(ffupper)
    
    
    
    
### pos ###
grid_shape_pos = [[left_x, right_x], [bottom_y, top_y]]
H = grid_shape_pos[1][1]-grid_shape_pos[1][0]
W = grid_shape_pos[0][1]-grid_shape_pos[0][0]
grid_size_pos = (int(41*W/H), 41)


steps = np.product(grid_size_pos)
A, B = grid_size_pos

field_pos = []
ff_pos = []
for en, n in enumerate(pick_neuron):
    covariates = [pref_hdw[en, len(w_arr)//2]*np.ones(steps), 
                  0.*np.ones(steps), 0.*np.ones(steps), 
                  np.linspace(left_x, right_x, A)[:, None].repeat(B, axis=1).flatten(), 
                  np.linspace(bottom_y, top_y, B)[None, :].repeat(A, axis=0).flatten(), 
                  t*np.ones(steps)]

    P_mc = model_utils.compute_P(full_model, covariates, [n], MC=MC_)[:, 0, ...].cpu()
    avg = (x_counts[None, None, :]*P_mc).sum(-1).reshape(-1, A, B).numpy()
    var = (x_counts[None, None, :]**2*P_mc).sum(-1).reshape(-1, A, B).numpy()
    xcvar = (var-avg**2)

    field_pos.append(avg.mean(0))
    ff_pos.append((xcvar/(avg+1e-12)).mean(0))


field_pos = np.stack(field_pos)
ff_pos = np.stack(ff_pos)




# compute the Pearson correlation between Fano factors and mean firing rates
b = 1
Pearson_ff = []
ratio = []
for avg, ff in zip(avg_models[b], ff_models[b]):
    r, r_p = scstats.pearsonr(ff, avg) # Pearson r correlation test
    Pearson_ff.append((r, r_p))
    ratio.append(ff.std()/avg.std())
    
    
    
data_run = (
    avg_models, var_models, ff_models, 
    Pearson_ff, ratio, 
    PLL_rg_ll, PLL_rg_cov, 
    Qq_ll, Zz_ll, R_ll, Rp_ll, q_DS_ll, T_DS_ll, T_KS_ll, 
    sign_KS, sign_DS, 
    mhd_mean, mhd_ff, hd_mean_tf, hd_ff_tf, 
    mw_mean, mw_ff, w_mean_tf, w_ff_tf, 
    ms_mean, ms_ff, s_mean_tf, s_ff_tf, 
    mt_mean, mt_ff, t_mean_tf, t_ff_tf, 
    mpos_mean, mpos_ff, pos_mean_tf, pos_ff_tf, 
    covariates_hd, lower_hd, mean_hd, upper_hd, 
    fflower_hd, ffmean_hd, ffupper_hd, 
    covariates_s, lower_s, mean_s, upper_s, 
    fflower_s, ffmean_s, ffupper_s, 
    covariates_t, lower_t, mean_t, upper_t, 
    fflower_t, ffmean_t, ffupper_t, 
    covariates_w, lower_w, mean_w, upper_w, 
    fflower_w, ffmean_w, ffupper_w, 
    grid_size_pos, grid_shape_pos, field_pos, ff_pos, 
    grid_size_hdw, grid_shape_hdw, field_hdw, 
    grid_size_hdt, grid_shape_hdt, field_hdt, 
    pref_hdw, ATI, res_var, 
    pref_hdt, drift, res_var_drift, 
    tun_width, amp_t, ampm_t, sim_mat, 
    pick_neuron, max_count, tbin, rcov, region_edge
)

pickle.dump(data_run, open('./saves/P_HDC_rg40.p', 'wb'))