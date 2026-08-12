# Combined, MEMORY-SAFE (streaming) regime analysis for large output.nc files.
# Computes in a single pass over time blocks, touching only river cells:
#   1. CFL / Courant: peak-discharge wave celerity vs cell length  -> cells/step
#   2. Quasi-steady ratio  r = |w*l*dh/dt| / Q  (aggregate + conditional on Q jump)
#   3. Flood-front check: does r stay small during big jumps on FLOWING cells?
#
# Percentiles for r are estimated from a fine log-spaced histogram (exact storage
# of ~10^8 samples is avoided). Per-cell CFL stats are exact (only ~10^4 cells).
#
# Usage: julia --project=. scripts/regime_analysis.jl <catchment_dir> [block]

using NCDatasets, Statistics, Printf

catch_dir = length(ARGS) >= 1 ? ARGS[1] : "models/sava_small_v081"
blockT    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
dt        = 86400.0
QWET      = 5.0   # m3/s: "meaningful flow" floor separating wet river from dry headwater

static_f  = joinpath(catch_dir, "staticmaps.nc")
output_f  = joinpath(catch_dir, "run_historical", "output.nc")

nan_if_missing(x) = ismissing(x) ? NaN : Float64(x)

# ---- static river geometry ---------------------------------------------------
sds = NCDataset(static_f, "r")
mask  = sds["wflow_river"][:, :]
len   = nan_if_missing.(sds["wflow_riverlength"][:, :])
width = nan_if_missing.(sds["wflow_riverwidth"][:, :])
slope = haskey(sds, "RiverSlope") ? nan_if_missing.(sds["RiverSlope"][:, :]) : fill(NaN, size(mask))
mann  = haskey(sds, "N_River")    ? nan_if_missing.(sds["N_River"][:, :])    : fill(0.036, size(mask))
close(sds)

nlon, nlat = size(mask)
isriver(r,c) = !ismissing(mask[r,c]) && mask[r,c] > 0
ridx = Int[]                                   # column-major linear indices of river cells
for c in 1:nlat, r in 1:nlon
    isriver(r,c) && push!(ridx, (c-1)*nlon + r)
end
ncell = length(ridx)
Lv = [len[i]   for i in ridx]
Wv = [width[i] for i in ridx]
Sv = [max(slope[i], 1e-5) for i in ridx]
Nv = [isnan(mann[i]) ? 0.036 : mann[i] for i in ridx]

# ---- histogram machinery for r percentiles -----------------------------------
# log-spaced bin edges for r in [1e-5, 1e5], plus under/overflow
const LR_LO, LR_HI, LR_STEP = -5.0, 5.0, 0.05
nrb = Int(round((LR_HI - LR_LO)/LR_STEP)) + 2      # +1 underflow, +1 overflow
rbin(r) = begin
    (isnan(r) || r <= 0) && return 1
    lr = log10(r)
    lr < LR_LO && return 1
    lr >= LR_HI && return nrb
    2 + Int(floor((lr - LR_LO)/LR_STEP))
end
rbin_center(b) = b == 1 ? 10.0^(LR_LO - LR_STEP/2) :
                 b == nrb ? 10.0^(LR_HI + LR_STEP/2) :
                 10.0^(LR_LO + (b-2+0.5)*LR_STEP)
function hist_pct(h, p)                              # percentile from cumulative histogram
    tot = sum(h); tot == 0 && return NaN
    target = p*tot; c = 0
    for b in 1:length(h)
        c += h[b]
        c >= target && return rbin_center(b)
    end
    rbin_center(length(h))
end

# jump bins on g = |dQ|/max(Q_t,Q_{t-1})
jedges = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-9]
jlabel = ["<5%","5-10%","10-20%","20-40%","40-60%","60-80%",">80%"]
njb = length(jlabel)
jbin(g) = begin
    for b in 1:njb
        (jedges[b] <= g < jedges[b+1]) && return b
    end
    njb
end

let
Hr = zeros(Int, njb, nrb)          # r-histogram per jump bin
rlt01 = 0; rlt03 = 0; ntot = 0     # aggregate fractions
# flood fronts on flowing cells (g>0.4 & Q>=QWET)
ff_n = 0; ff_lt01 = 0; ff_lt03 = 0; ff_maxr = 0.0
Hff = zeros(Int, nrb)
# wet-vs-dry split of the worst jump bin (g>0.8)
worst_wet_n = 0; worst_dry_n = 0; worst_wet_maxr = 0.0; worst_dry_maxr = 0.0

# ---- per-cell peak accumulators (exact CFL) ----------------------------------
qpeak = fill(-Inf, ncell)
hpeak = fill(-Inf, ncell)

# ---- streaming pass over time blocks -----------------------------------------
ods = NCDataset(output_f, "r")
qvar = ods["river_q"]; hvar = ods["river_h"]
nt = size(qvar, 3)
prevh = fill(NaN, ncell); prevq = fill(NaN, ncell)
t0 = 1
@printf("streaming %s : %d river cells, %d steps, block=%d\n", catch_dir, ncell, nt, blockT)
while t0 <= nt
    t1 = min(t0 + blockT - 1, nt)
    qb = reshape(Array(qvar[:, :, t0:t1]), nlon*nlat, t1-t0+1)
    hb = reshape(Array(hvar[:, :, t0:t1]), nlon*nlat, t1-t0+1)
    qc = @view qb[ridx, :]
    hc = @view hb[ridx, :]
    Tb = t1 - t0 + 1
    for t in 1:Tb
        for i in 1:ncell
            q = nan_if_missing(qc[i, t]); h = nan_if_missing(hc[i, t])
            if !isnan(q) && q > qpeak[i]; qpeak[i] = q; end
            if !isnan(h) && h > hpeak[i]; hpeak[i] = h; end
            ph = prevh[i]; pq = prevq[i]
            if !isnan(q) && !isnan(h) && !isnan(ph) && !isnan(pq)
                w = Wv[i]; l = Lv[i]
                if !isnan(w) && !isnan(l) && w > 0 && l > 0
                    qbig = max(q, pq)
                    if qbig > 1e-6
                        g = abs(q - pq)/qbig
                        dS = w*l*(h - ph)/dt
                        r  = abs(dS)/max(q, 1e-6)
                        jb = jbin(g); rb = rbin(r)
                        Hr[jb, rb] += 1
                        ntot += 1
                        r < 0.1 && (rlt01 += 1)
                        r < 0.3 && (rlt03 += 1)
                        if g > 0.4 && q >= QWET
                            ff_n += 1; Hff[rb] += 1
                            r < 0.1 && (ff_lt01 += 1)
                            r < 0.3 && (ff_lt03 += 1)
                            r > ff_maxr && (ff_maxr = r)
                        end
                        if g > 0.8
                            if q >= QWET
                                worst_wet_n += 1; r > worst_wet_maxr && (worst_wet_maxr = r)
                            else
                                worst_dry_n += 1; r > worst_dry_maxr && (worst_dry_maxr = r)
                            end
                        end
                    end
                end
            end
            prevh[i] = h; prevq[i] = q
        end
    end
    @printf("  ... steps %d-%d done\n", t0, t1)
    t0 = t1 + 1
end
close(ods)

# ---- CFL stats (exact over cells) --------------------------------------------
good = @. isfinite(qpeak) & (qpeak > 0) & isfinite(hpeak) & !isnan(Lv) & !isnan(Wv) & (Lv > 0) & (Wv > 0)
Lg, Wg, Sg, Ng = Lv[good], Wv[good], Sv[good], Nv[good]
Qp, Hp = qpeak[good], hpeak[good]
Hc = @. max(Hp, 0.01)
v_cont = @. Qp/(Wg*Hc)
v_mann = @. (1.0/Ng)*Hc^(2/3)*sqrt(Sg)
c_cont = (5/3).*v_cont
c_mann = (5/3).*v_mann
Cr_cont = @. c_cont*dt/Lg
Cr_mann = @. c_mann*dt/Lg
pc(x,p) = quantile(filter(isfinite, x), p)

@printf("\n=== CFL / information travel: %s ===\n", catch_dir)
@printf("river cells (peak Q>0)        : %d\n", length(Lg))
@printf("river_length [m] median %.0f (p10 %.0f, p90 %.0f)\n", median(Lg), pc(Lg,0.1), pc(Lg,0.9))
@printf("river_width  [m] median %.1f (p10 %.1f, p90 %.1f)\n", median(Wg), pc(Wg,0.1), pc(Wg,0.9))
@printf("peak Q [m3/s]    median %.1f (p90 %.1f, max %.1f)\n", median(Qp), pc(Qp,0.9), maximum(Qp))
@printf("peak depth [m]   median %.2f (p90 %.2f, max %.2f)\n", median(Hp), pc(Hp,0.9), maximum(Hp))
@printf("celerity c=(5/3)Q/(wh) [m/s] median %.2f (p90 %.2f, max %.2f)\n", median(c_cont), pc(c_cont,0.9), maximum(c_cont))
@printf("travel c*dt [km]  median %.1f (p90 %.1f, max %.1f)\n", median(c_cont.*dt)/1e3, pc(c_cont.*dt,0.9)/1e3, maximum(c_cont.*dt)/1e3)
@printf("Courant Cr=c*dt/L cont   median %.1f (p90 %.1f, p99 %.1f, max %.1f)\n", median(Cr_cont), pc(Cr_cont,0.9), pc(Cr_cont,0.99), maximum(Cr_cont))
@printf("Courant Cr        manning median %.1f (p90 %.1f, p99 %.1f, max %.1f)\n", median(Cr_mann), pc(Cr_mann,0.9), pc(Cr_mann,0.99), maximum(Cr_mann))
@printf("fraction cells Cr>1: cont %.0f%%, manning %.0f%%\n", 100*count(>(1),Cr_cont)/length(Cr_cont), 100*count(>(1),Cr_mann)/length(Cr_mann))

@printf("\n=== quasi-steady ratio r = |w*l*dh/dt|/Q : %s ===\n", catch_dir)
@printf("samples (cell x step): %d\n", ntot)
@printf("aggregate r: median~%.4f  fraction r<0.1 %.1f%%  r<0.3 %.1f%%\n",
        hist_pct(vec(sum(Hr, dims=1)), 0.5), 100*rlt01/ntot, 100*rlt03/ntot)
@printf("\n%-8s %10s | %10s %10s %10s\n", "Qjump", "n", "r_med~", "r_p90~", "r_p99~")
for b in 1:njb
    row = Hr[b, :]; n = sum(row); n == 0 && continue
    @printf("%-8s %10d | %10.4f %10.4f %10.4f\n", jlabel[b], n,
            hist_pct(row,0.5), hist_pct(row,0.9), hist_pct(row,0.99))
end
@printf("\n-- flood fronts on flowing cells (g>0.4 AND Q>=%.0f m3/s) --\n", QWET)
if ff_n > 0
    @printf("n=%d  r_med~%.4f p90~%.4f p99~%.4f  max %.3f  | r<0.1 %.1f%%  r<0.3 %.1f%%\n",
            ff_n, hist_pct(Hff,0.5), hist_pct(Hff,0.9), hist_pct(Hff,0.99), ff_maxr,
            100*ff_lt01/ff_n, 100*ff_lt03/ff_n)
else
    println("no wet flood-front samples")
end
@printf("\n-- worst jump bin (g>0.8) wet-vs-dry split --\n")
@printf("wet (Q>=%.0f): n=%d  max r %.3f\n", QWET, worst_wet_n, worst_wet_maxr)
@printf("dry (Q< %.0f): n=%d  max r %.3e\n", QWET, worst_dry_n, worst_dry_maxr)
end
