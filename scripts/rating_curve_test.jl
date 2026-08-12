# Empirical validity of the Manning rating curve  Q = (1/n) w H^(5/3) sqrt(S)
# as a constraint on river discharge.  Tests, per (cell, step), the ratio
#     rho = Q_manning / Q_actual
# against the wflow_sbm reference run, broken down by the conditions that are
# supposed to govern its validity:
#   - flow magnitude (dry vs meaningful flow)
#   - quasi-steadiness           r = |w l dh/dt| / Q   (kinematic-wave regime)
#   - in-bank vs overbank        H <= RiverDepth (bankfull)  vs  H > RiverDepth
# Also reports the rectangular-section form (R = wH/(w+2H)) as a secondary check.
#
# Streaming (block over time) so it runs on the 11.6 GB sava_v081 output.
# Usage: julia --project=. scripts/rating_curve_test.jl <catchment_dir> [block]

using NCDatasets, Statistics, Printf

function main()
    catch_dir = length(ARGS) >= 1 ? ARGS[1] : "models/sava_small_v081"
    blockT    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    dt        = 86400.0
    QWET      = 5.0

    static_f  = joinpath(catch_dir, "staticmaps.nc")
    output_f  = joinpath(catch_dir, "run_historical", "output.nc")
    nanm(x) = ismissing(x) ? NaN : Float64(x)

    sds = NCDataset(static_f, "r")
    mask  = sds["wflow_river"][:, :]
    len   = nanm.(sds["wflow_riverlength"][:, :])
    width = nanm.(sds["wflow_riverwidth"][:, :])
    slope = haskey(sds, "RiverSlope") ? nanm.(sds["RiverSlope"][:, :]) : fill(NaN, size(mask))
    mann  = haskey(sds, "N_River")    ? nanm.(sds["N_River"][:, :])    : fill(0.036, size(mask))
    dbf   = haskey(sds, "RiverDepth") ? nanm.(sds["RiverDepth"][:, :])  : fill(NaN, size(mask))
    close(sds)

    nlon, nlat = size(mask)
    isriver(r,c) = !ismissing(mask[r,c]) && mask[r,c] > 0
    ridx = Int[]
    for c in 1:nlat, r in 1:nlon
        isriver(r,c) && push!(ridx, (c-1)*nlon + r)
    end
    ncell = length(ridx)
    Lv = [len[i]   for i in ridx]
    Wv = [width[i] for i in ridx]
    Sv = [max(slope[i], 1e-6) for i in ridx]
    Nv = [isnan(mann[i]) || mann[i] <= 0 ? 0.036 : mann[i] for i in ridx]
    Dv = [dbf[i] for i in ridx]

    # ---- log-spaced histogram for rho = Q_man/Q_act -------------------------
    LR_LO, LR_HI, LR_STEP = -3.0, 3.0, 0.02
    nrb = Int(round((LR_HI-LR_LO)/LR_STEP)) + 2
    rbin(x) = begin
        (isnan(x) || x <= 0) && return 1
        lx = log10(x)
        lx < LR_LO && return 1
        lx >= LR_HI && return nrb
        2 + Int(floor((lx-LR_LO)/LR_STEP))
    end
    rcen(b) = b==1 ? 10.0^(LR_LO-LR_STEP/2) : b==nrb ? 10.0^(LR_HI+LR_STEP/2) :
              10.0^(LR_LO+(b-2+0.5)*LR_STEP)
    hpct(h,p) = begin
        tot=sum(h); tot==0 && return NaN
        t=p*tot; c=0; res=rcen(length(h))
        for b in 1:length(h); c+=h[b]; if c>=t; res=rcen(b); break; end; end
        res
    end
    # within-tolerance counters use exact ratio, not histogram
    labels = ["ALL", "wet Q>=5", "quasi-steady r<0.1", "unsteady r>=0.1",
              "in-bank H<=Dbf", "overbank H>Dbf",
              "flow 5-50", "flow 50-500", "flow >500"]
    ncond = length(labels)
    Hw = [zeros(Int, nrb) for _ in 1:ncond]   # wide-channel rho histogram per condition
    within25 = zeros(Int, ncond); within50 = zeros(Int, ncond); ncnt = zeros(Int, ncond)
    # rectangular form, ALL only
    Hrect = zeros(Int, nrb); rect_w25 = 0; rect_n = 0

    ods = NCDataset(output_f, "r")
    qvar = ods["river_q"]; hvar = ods["river_h"]
    nt = size(qvar, 3)
    prevh = fill(NaN, ncell)
    @printf("streaming %s : %d river cells, %d steps\n", catch_dir, ncell, nt)
    t0 = 1
    while t0 <= nt
        t1 = min(t0+blockT-1, nt)
        qb = reshape(Array(qvar[:, :, t0:t1]), nlon*nlat, t1-t0+1)
        hb = reshape(Array(hvar[:, :, t0:t1]), nlon*nlat, t1-t0+1)
        qc = @view qb[ridx, :]; hc = @view hb[ridx, :]
        Tb = t1-t0+1
        for t in 1:Tb
            for i in 1:ncell
                q = nanm(qc[i,t]); h = nanm(hc[i,t])
                ph = prevh[i]; prevh[i] = h
                (isnan(q) || isnan(h) || q <= 0 || h <= 0) && continue
                w = Wv[i]; l = Lv[i]; S = Sv[i]; n = Nv[i]; Dbf = Dv[i]
                (isnan(w) || isnan(l) || w <= 0) && continue
                q_wide = (1.0/n) * w * h^(5/3) * sqrt(S)
                R      = w*h/(w + 2h)
                q_rect = (1.0/n) * (w*h) * R^(2/3) * sqrt(S)
                rho  = q_wide / q
                rhor = q_rect / q
                # r (quasi-steady) needs previous h
                r = isnan(ph) ? NaN : abs(w*l*(h-ph)/dt)/max(q,1e-6)
                bw = rbin(rho)
                acc!(cond) = begin
                    Hw[cond][bw] += 1; ncnt[cond] += 1
                    (0.8 <= rho <= 1.25) && (within25[cond] += 1)
                    (0.5 <= rho <= 2.0)  && (within50[cond] += 1)
                end
                acc!(1)                                   # ALL
                q >= QWET && acc!(2)
                if !isnan(r)
                    r < 0.1 ? acc!(3) : acc!(4)
                end
                if !isnan(Dbf) && Dbf > 0
                    h <= Dbf ? acc!(5) : acc!(6)
                end
                if q >= 5 && q < 50; acc!(7)
                elseif q >= 50 && q < 500; acc!(8)
                elseif q >= 500; acc!(9); end
                # rectangular, ALL only
                Hrect[rbin(rhor)] += 1; rect_n += 1
                (0.8 <= rhor <= 1.25) && (rect_w25 += 1)
            end
        end
        @printf("  ... steps %d-%d\n", t0, t1)
        t0 = t1+1
    end
    close(ods)

    @printf("\n=== Manning rating-curve validity: %s ===\n", catch_dir)
    @printf("rho = Q_manning(wide) / Q_actual ;  rating 'valid' ~ rho near 1\n\n")
    @printf("%-22s %10s | %9s %9s %9s | %9s %9s\n",
            "condition","n","rho_p10","rho_med","rho_p90","within25%","within50%")
    for c in 1:ncond
        n = ncnt[c]; n == 0 && continue
        @printf("%-22s %10d | %9.3f %9.3f %9.3f | %8.1f%% %8.1f%%\n",
                labels[c], n, hpct(Hw[c],0.1), hpct(Hw[c],0.5), hpct(Hw[c],0.9),
                100*within25[c]/n, 100*within50[c]/n)
    end
    @printf("\nrectangular section form (R=wH/(w+2H)), ALL: rho_med %.3f  within25%% %.1f%%\n",
            hpct(Hrect,0.5), 100*rect_w25/max(rect_n,1))
end

main()
