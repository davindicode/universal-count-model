binsize = 100
rcov_lvm, neurons, tbin, resamples, rc_t, _ = HDC.get_dataset(session_id, phase, binsize, '../scripts/data')
max_count = int(rc_t.max())
rhd_t = rcov_lvm[0]


modes = [('GP', 'U', 'T1', 8, 'identity', 3, [0], False, 10, False, 'ew'), 
         ('GP', 'IP', 'T1', 8, 'exp', 1, [0], False, 10, False, 'ew'), 
         ('GP', 'hNB', 'T1', 8, 'exp', 1, [0], False, 10, False, 'ew')]



# likelihood CV over subgroups of neurons as well as validation runs
beta = 0.0
n_group = np.arange(5)
ncvx = 2
val_neuron = [n_group, n_group+5, n_group+10, n_group+15, n_group+20, n_group+25, np.arange(3)+30]
kcvs = [1, 2, 3, 5, 6, 8] # validation segments from splitting data into 10

LVM_cv_ll = []
for kcv in kcvs:
    for mode in modes:
        cvdata = model_utils.get_cv_sets(mode, [kcv], 5000, rc_t, resamples, rcov_lvm)[0]
        _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
        cv_set = (ftrain, fcov, vtrain, vcov)

        for v_neuron in val_neuron:
            fac = len(n_group)/len(v_neuron)

            prev_ll = np.inf
            for tr in range(ncvx):
                full_model = get_full_model(session_id, phase, cvdata, resamples, 100, 
                                            mode, rcov_lvm, max_count, neurons, gpu=gpu_dev)
                mask = np.ones((neurons,), dtype=bool)
                mask[v_neuron] = False
                f_neuron = np.arange(neurons)[mask]
                ll = model_utils.LVM_pred_ll(full_model, mode[-5], mode[2], models.cov_used, cv_set, f_neuron, v_neuron, 
                                             beta=beta, beta_z=0.0, max_iters=3000)[0]
                if ll < prev_ll:
                    prev_ll = ll

            LVM_cv_ll.append(fac*prev_ll)
        
LVM_cv_ll = np.array(LVM_cv_ll).reshape(len(kcvs), len(modes), len(val_neuron))




def circ_drift_regression(x, z, t, topology, dev='cpu', iters=1000, lr=1e-2, a_fac=1):
    t = torch.tensor(t, device=dev)
    X = torch.tensor(x, device=dev)
    Z = torch.tensor(z, device=dev)
        
    lowest_loss = np.inf
    for sign in [1, -1]: # select sign automatically
        shift = Parameter(torch.zeros(1, device=dev))
        a = Parameter(torch.zeros(1, device=dev))

        optimizer = optim.Adam([a, shift], lr=lr)
        losses = []
        for k in range(iters):
            optimizer.zero_grad()
            Z_ = t*a_fac*a + shift + sign*Z
            loss = (utils.latent.metric(Z_, X, topology)**2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().item())

        l_ = loss.cpu().item()
        
        if l_ < lowest_loss:
            lowest_loss = l_
            a_ = a.cpu().item()
            shift_ = shift.cpu().item()
            sign_ = sign
            losses_ = losses

    return a_fac*a_, sign_, shift_, losses_



# trajectory regression to align to data and compute drifts
topology = 'torus'
cvK = 3
CV = [0, 1, 2]

RMS_cv = []
drifts_lv = []
for mode in modes:
    cvdata = model_utils.get_cv_sets(mode, [-1], 5000, rc_t, resamples, rcov_lvm)[0]
    _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
    cv_set = (ftrain, fcov, vtrain, vcov)
        
    full_model = get_full_model(session_id, phase, cvdata, resamples, 100, 
                                mode, rcov_lvm, max_count, neurons, gpu=gpu_dev)

    X_loc, X_std = full_model.inputs.eval_XZ()
    cvT = X_loc[0].shape[0]
    tar_t = rhd_t[:cvT]
    lat = X_loc[0]
    
    for rn in CV:
        fit_range = np.arange(cvT//cvK) + rn*cvT//cvK

        drift, sign, shift, losses = circ_drift_regression(tar_t[fit_range], lat[fit_range], fit_range*tbin, 
                                                      topology, dev=dev, a_fac=1e-5)
        
        #plt.plot(losses)
        #plt.show()
        mask = np.ones((cvT,), dtype=bool)
        mask[fit_range] = False
        
        lat_t = torch.tensor((np.arange(cvT)*tbin*drift + shift + sign*lat) % (2*np.pi))
        D = (utils.latent.metric(torch.tensor(tar_t)[mask], lat_t[mask], topology)**2)
        RMS_cv.append(D.mean().item())
        drifts_lv.append(drift)


RMS_cv = np.array(RMS_cv).reshape(len(modes), len(CV))
drifts_lv = np.array(drifts_lv).reshape(len(modes), len(CV))



# compute delays in latent trajectory w.r.t. data, see which one fits best in RMS
topology = 'torus'
cvK = 3
CV = [0, 1, 2]

D = 5
delays = np.arange(-D, D+1)
delay_RMS = []
mode = modes[0]

for delay in delays:
    cvdata = model_utils.get_cv_sets(mode, [-1], 5000, rc_t, resamples, rcov_lvm)[0]
    _, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
    cv_set = (ftrain, fcov, vtrain, vcov)
        
    full_model = get_full_model(session_id, phase, cvdata, resamples, 100, 
                                mode, rcov_lvm, max_count, neurons, gpu=gpu_dev)

    X_loc, X_std = full_model.inputs.eval_XZ()
    cvT = X_loc[0].shape[0]-len(delays)+1
    tar_t = rhd_t[D+delay:cvT+D+delay]
    lat = X_loc[0][D:cvT+D]
    
    for rn in CV:
        fit_range = np.arange(cvT//cvK) + rn*cvT//cvK

        drift, sign, shift, _ = circ_drift_regression(tar_t[fit_range], lat[fit_range], fit_range*tbin, 
                                                      topology, dev=dev, a_fac=1e-5)
        
        mask = np.ones((cvT,), dtype=bool)
        mask[fit_range] = False
        
        lat_ = torch.tensor((np.arange(cvT)*tbin*drift + shift + sign*lat) % (2*np.pi))
        Dd = (utils.latent.metric(torch.tensor(tar_t)[mask], lat_[mask], topology)**2)
        delay_RMS.append(Dd.mean().item())


delay_RMS = np.array(delay_RMS).reshape(len(delays), len(CV))



# get the latent inferred trajectory
mode = modes[0]
topology = 'torus'


cvdata = model_utils.get_cv_sets(mode, [-1], 5000, rc_t, resamples, rcov_lvm)[0]
_, ftrain, fcov, vtrain, vcov, cvbatch_size = cvdata
cv_set = (ftrain, fcov, vtrain, vcov)

full_model = get_full_model(session_id, phase, cvdata, resamples, 100, 
                            mode, rcov_lvm, max_count, neurons, gpu=gpu_dev)

X_loc, X_std = full_model.inputs.eval_XZ()

tar_t = rhd_t
lat = X_loc[0]

drift, sign, shift, _ = circ_drift_regression(tar_t[fit_range], lat[fit_range], fit_range*tbin, 
                                              topology, dev=dev, a_fac=1e-5)

lat_t = ((np.arange(rhd_t.shape[0])*tbin*drift + shift + sign*lat) % (2*np.pi))
lat_t_std = X_std[0]



data_run = (
    lat_t, lat_t_std, delay_RMS, RMS_cv, LVM_cv_ll, drifts_lv, rcov_lvm
)

pickle.dump(data_run, open('./saves/P_HDC_lat.p', 'wb'))