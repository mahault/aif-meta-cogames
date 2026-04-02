#!/usr/bin/env python3
"""Analyze learned generative model parameters vs hand-crafted defaults."""
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aif_meta_cogames.aif_agent.generative_model import (
    A_DEPENDENCIES, B_DEPENDENCIES, NUM_STATE_FACTORS,
    build_default_A, build_option_B,
    build_C_miner, build_C_aligner, build_C_scout,
)
from aif_meta_cogames.aif_agent.discretizer import (
    NUM_OBS, NUM_OPTIONS, OPTION_NAMES, OBS_MODALITY_NAMES,
    Phase, Hand, TargetMode, Role,
    ObsResource, ObsStation, ObsInventory, ObsContest, ObsSocial, ObsRoleSignal,
)

SFN = ["phase", "hand", "target_mode", "role"]
SFL = {
    0: [p.name for p in Phase],
    1: [h.name for h in Hand],
    2: [t.name for t in TargetMode],
    3: [r.name for r in Role],
}
OL = {
    0: [o.name for o in ObsResource],
    1: [o.name for o in ObsStation],
    2: [o.name for o in ObsInventory],
    3: [o.name for o in ObsContest],
    4: [o.name for o in ObsSocial],
    5: [o.name for o in ObsRoleSignal],
}


def analyze_A(Ad, Al, tk=10):
    print("=" * 80)
    print("  A MATRIX ANALYSIS (Observation Likelihood)")
    print("=" * 80)
    tc, te = 0, 0
    for m in range(len(Ad)):
        ad = np.asarray(Ad[m], dtype=np.float64)
        al = np.asarray(Al[m], dtype=np.float64)
        diff = al - ad
        absd = np.abs(diff)
        ne = diff.size
        te += ne
        sc = int(np.sum(absd > 0.05))
        tc += sc
        deps = A_DEPENDENCIES[m]
        dn = [SFN[f] for f in deps]
        print()
        print("-" * 70)
        print("A[%d]: %s | shape %s" % (m, OBS_MODALITY_NAMES[m], ad.shape))
        print("  deps: %s  mean|d|=%.4f max|d|=%.4f RMSE=%.4f sig>0.05: %d/%d" % (
            dn, absd.mean(), absd.max(), np.sqrt(np.mean(diff**2)), sc, ne))
        fi = np.argsort(absd.ravel())[::-1][:tk]
        print()
        print("  Top-%d:" % tk)
        print("  %-20s %-35s %8s %8s %8s" % ("Obs", "State", "Def", "Lrn", "Delta"))
        print("  " + "-" * 20 + " " + "-" * 35 + " " + "-" * 8 + " " + "-" * 8 + " " + "-" * 8)
        for idx in fi:
            mi = np.unravel_index(idx, ad.shape)
            ol2 = OL[m][mi[0]]
            sp = []
            for i, f in enumerate(deps):
                sp.append("%s=%s" % (SFN[f], SFL[f][mi[1 + i]]))
            sl = ", ".join(sp)
            print("  %-20s %-35s %8.4f %8.4f %+8.4f %s" % (
                ol2, sl, ad[mi], al[mi], diff[mi], "^" if diff[mi] > 0 else "v"))
        # Full matrix for small modalities
        if ad.ndim == 2 and ad.shape[0] <= 4 and ad.shape[1] <= 6:
            sl2 = SFL[deps[0]]
            print()
            print("  Full matrix (def / lrn / delta):")
            h = "  %18s" % ""
            for s in range(ad.shape[1]):
                h += " %12s" % sl2[s]
            print(h)
            for o in range(ad.shape[0]):
                ol2 = OL[m][o]
                ld = "  def %-12s" % ol2
                ll = "  lrn %-12s" % ol2
                lx = "   d  %-12s" % ol2
                for s in range(ad.shape[1]):
                    ld += " %12.4f" % ad[o, s]
                    ll += " %12.4f" % al[o, s]
                    d = diff[o, s]
                    mk = "*" if abs(d) > 0.05 else " "
                    lx += " %+11.4f%s" % (d, mk)
                print(ld)
                print(ll)
                print(lx)
                print()
    print()
    print("=" * 70)
    print("A SUMMARY: %d entries changed >0.05 / %d total" % (tc, te))


def analyze_B(Bd, Bl, tk=10):
    print()
    print("=" * 80)
    print("  B MATRIX ANALYSIS (Transition Model, 5 Macro-Options)")
    print("=" * 80)
    tc, te = 0, 0
    for f in range(len(Bd)):
        bd = np.asarray(Bd[f], dtype=np.float64)
        bl = np.asarray(Bl[f], dtype=np.float64)
        diff = bl - bd
        absd = np.abs(diff)
        ne = diff.size
        te += ne
        sc = int(np.sum(absd > 0.05))
        tc += sc
        deps = B_DEPENDENCIES[f]
        dn = [SFN[d] for d in deps]
        print()
        print("-" * 70)
        print("B[%d]: %s | shape %s" % (f, SFN[f], bd.shape))
        print("  deps: %s  mean|d|=%.4f max|d|=%.4f RMSE=%.4f sig>0.05: %d/%d" % (
            dn, absd.mean(), absd.max(), np.sqrt(np.mean(diff**2)), sc, ne))
        fi = np.argsort(absd.ravel())[::-1][:tk]
        print()
        print("  Top-%d:" % tk)
        if bd.ndim == 4:
            print("  %-18s %-30s %-15s %8s %8s %8s" % (
                "Next", "Cur deps", "Option", "Def", "Lrn", "Delta"))
            print("  " + "-" * 18 + " " + "-" * 30 + " " + "-" * 15 +
                  " " + "-" * 8 + " " + "-" * 8 + " " + "-" * 8)
            for idx in fi:
                mi = np.unravel_index(idx, bd.shape)
                nl = SFL[f][mi[0]]
                cp = []
                for i, d in enumerate(deps):
                    cp.append("%s=%s" % (SFN[d], SFL[d][mi[1 + i]]))
                cl = ", ".join(cp)
                ol = OPTION_NAMES[mi[-1]]
                print("  %-18s %-30s %-15s %8.4f %8.4f %+8.4f %s" % (
                    nl, cl, ol, bd[mi], bl[mi], diff[mi],
                    "^" if diff[mi] > 0 else "v"))
        elif bd.ndim == 3:
            print("  %-15s %-15s %-15s %8s %8s %8s" % (
                "Next", "Current", "Option", "Def", "Lrn", "Delta"))
            print("  " + "-" * 15 + " " + "-" * 15 + " " + "-" * 15 +
                  " " + "-" * 8 + " " + "-" * 8 + " " + "-" * 8)
            for idx in fi:
                mi = np.unravel_index(idx, bd.shape)
                nl = SFL[f][mi[0]]
                cl = SFL[f][mi[1]]
                ol = OPTION_NAMES[mi[2]]
                print("  %-15s %-15s %-15s %8.4f %8.4f %+8.4f %s" % (
                    nl, cl, ol, bd[mi], bl[mi], diff[mi],
                    "^" if diff[mi] > 0 else "v"))
        print()
        print("  Per-option RMSE:")
        for o in range(bd.shape[-1]):
            do = diff[..., o] if bd.ndim == 3 else diff[:, :, :, o]
            print("    %-15s RMSE=%.4f max=%.4f" % (
                OPTION_NAMES[o], np.sqrt(np.mean(do**2)), np.abs(do).max()))
    print()
    print("=" * 70)
    print("B SUMMARY: %d entries changed >0.05 / %d total" % (tc, te))


def B_detail(Bd, Bl):
    print()
    print("=" * 80)
    print("  B TRANSITION DETAIL (Phase, hand=EMPTY, per option)")
    print("=" * 80)
    bd = np.asarray(Bd[0], dtype=np.float64)
    bl = np.asarray(Bl[0], dtype=np.float64)
    pl = SFL[0]
    for o in range(len(OPTION_NAMES)):
        td = bd[:, :, 0, o]
        tl = bl[:, :, 0, o]
        rmse = np.sqrt(np.mean((tl - td)**2))
        if rmse < 0.01:
            continue
        print()
        print("  %s (hand=EMPTY) RMSE=%.4f" % (OPTION_NAMES[o], rmse))
        h = "  %12s" % ""
        for p in range(len(pl)):
            h += " %8s" % pl[p]
        print("  DEFAULT:")
        print(h)
        for pn in range(len(pl)):
            l = "  to %8s" % pl[pn]
            for pc in range(len(pl)):
                l += " %8.3f" % td[pn, pc]
            print(l)
        print("  LEARNED:")
        print(h)
        for pn in range(len(pl)):
            l = "  to %8s" % pl[pn]
            for pc in range(len(pl)):
                l += " %8.3f" % tl[pn, pc]
            print(l)
        print("  DELTA:")
        print(h)
        for pn in range(len(pl)):
            l = "  to %8s" % pl[pn]
            for pc in range(len(pl)):
                d = tl[pn, pc] - td[pn, pc]
                mk = "*" if abs(d) > 0.05 else " "
                l += " %+7.3f%s" % (d, mk)
            print(l)


def analyze_C(rd):
    print()
    print("=" * 80)
    print("  C VECTOR ANALYSIS (Preferences per Role)")
    print("=" * 80)
    dc = {
        "miner": build_C_miner(),
        "aligner": build_C_aligner(),
        "scout": build_C_scout(),
    }
    for role in ("miner", "aligner", "scout"):
        print()
        print("-" * 70)
        print("  Role: %s" % role.upper())
        for m in range(6):
            cd = np.asarray(dc[role][m], dtype=np.float64)
            cl = np.asarray(rd["C_%s_%d" % (role, m)], dtype=np.float64)
            diff = cl - cd
            ol = OL[m]
            print()
            print("  %s (%d obs):" % (OBS_MODALITY_NAMES[m], len(ol)))
            print("    %-20s %8s %8s %8s" % ("Obs", "Def", "Lrn", "Delta"))
            print("    " + "-" * 20 + " " + "-" * 8 + " " + "-" * 8 + " " + "-" * 8)
            for o in range(len(ol)):
                d = diff[o]
                mk = " **" if abs(d) > 0.5 else (" *" if abs(d) > 0.1 else "")
                print("    %-20s %8.3f %8.3f %+8.3f%s" % (
                    ol[o], cd[o], cl[o], d, mk))
            do = np.argsort(-cd)
            lo = np.argsort(-cl)
            dr = [ol[i] for i in do]
            lr = [ol[i] for i in lo]
            ch = " <-- REORDERED" if dr != lr else ""
            print("    Def rank: %s" % " > ".join(dr))
            print("    Lrn rank: %s%s" % (" > ".join(lr), ch))


def kl_analysis(Ad, Al, Bd, Bl):
    print()
    print("=" * 80)
    print("  KL DIVERGENCE: D_KL[learned || default]")
    print("=" * 80)
    eps = 1e-10
    print()
    print("  A matrices:")
    tka = 0.0
    for m in range(len(Ad)):
        ad = np.clip(np.asarray(Ad[m], dtype=np.float64).ravel(), eps, 1.0)
        al = np.clip(np.asarray(Al[m], dtype=np.float64).ravel(), eps, 1.0)
        kl = np.sum(al * np.log(al / ad))
        tka += kl
        print("    A[%d] %-18s KL=%.4f nats" % (m, OBS_MODALITY_NAMES[m], kl))
    print("    Total A KL: %.4f" % tka)
    print()
    print("  B matrices:")
    tkb = 0.0
    for f in range(len(Bd)):
        bd = np.clip(np.asarray(Bd[f], dtype=np.float64).ravel(), eps, 1.0)
        bl = np.clip(np.asarray(Bl[f], dtype=np.float64).ravel(), eps, 1.0)
        kl = np.sum(bl * np.log(bl / bd))
        tkb += kl
        print("    B[%d] %-18s KL=%.4f nats" % (f, SFN[f], kl))
    print("    Total B KL: %.4f" % tkb)
    print()
    print("    Grand total: %.4f nats (A=%.1f, B=%.1f)" % (tka + tkb, tka, tkb))


def entropy_analysis(Ad, Al, Bd, Bl):
    print()
    print("=" * 80)
    print("  ENTROPY: sharper or more diffuse?")
    print("=" * 80)
    eps = 1e-10
    def ce(matrix):
        s = matrix.shape
        m2 = matrix.reshape(s[0], -1)
        m2 = np.clip(m2, eps, 1.0)
        m2 = m2 / m2.sum(axis=0, keepdims=True)
        return (-np.sum(m2 * np.log(m2), axis=0)).mean()
    print()
    print("  A matrices:")
    print("    %-24s %10s %10s %10s %s" % ("Modality", "H_def", "H_lrn", "Delta", ""))
    for m in range(len(Ad)):
        hd = ce(np.asarray(Ad[m]))
        hl = ce(np.asarray(Al[m]))
        d = hl - hd
        i = "sharper" if d < -0.01 else ("diffuse" if d > 0.01 else "similar")
        print("    %-24s %10.4f %10.4f %+10.4f %s" % (
            OBS_MODALITY_NAMES[m], hd, hl, d, i))
    print()
    print("  B matrices:")
    print("    %-24s %10s %10s %10s %s" % ("Factor", "H_def", "H_lrn", "Delta", ""))
    for f in range(len(Bd)):
        hd = ce(np.asarray(Bd[f]))
        hl = ce(np.asarray(Bl[f]))
        d = hl - hd
        i = "sharper" if d < -0.01 else ("diffuse" if d > 0.01 else "similar")
        print("    %-24s %10.4f %10.4f %+10.4f %s" % (SFN[f], hd, hl, d, i))


def do_summary(Ad, Al, Bd, Bl, rd, meta):
    print()
    print("=" * 80)
    print("  OVERALL SUMMARY")
    print("=" * 80)
    if meta:
        print()
        print("  Metadata: %s" % meta)
    print()
    print("  A changes:")
    ar = []
    for m in range(len(Ad)):
        d = np.asarray(Al[m]) - np.asarray(Ad[m])
        r = np.sqrt(np.mean(d**2))
        ar.append((r, m))
        print("    A[%d] %-18s RMSE=%.4f max|d|=%.4f" % (
            m, OBS_MODALITY_NAMES[m], r, np.abs(d).max()))
    ar.sort(reverse=True)
    print("  Most changed: %s (RMSE=%.4f)" % (OBS_MODALITY_NAMES[ar[0][1]], ar[0][0]))
    print()
    print("  B changes:")
    br = []
    for f in range(len(Bd)):
        d = np.asarray(Bl[f]) - np.asarray(Bd[f])
        r = np.sqrt(np.mean(d**2))
        br.append((r, f))
        print("    B[%d] %-18s RMSE=%.4f max|d|=%.4f" % (f, SFN[f], r, np.abs(d).max()))
    br.sort(reverse=True)
    print("  Most changed: %s (RMSE=%.4f)" % (SFN[br[0][1]], br[0][0]))
    dc = {"miner": build_C_miner(), "aligner": build_C_aligner(), "scout": build_C_scout()}
    print()
    print("  C changes:")
    for role in ("miner", "aligner", "scout"):
        md, mm, mo = 0.0, "", ""
        for mi in range(6):
            cd = np.asarray(dc[role][mi])
            cl = np.asarray(rd["C_%s_%d" % (role, mi)])
            d = cl - cd
            ix = np.argmax(np.abs(d))
            if abs(d[ix]) > abs(md):
                md = d[ix]
                mm = OBS_MODALITY_NAMES[mi]
                mo = OL[mi][ix]
        dr = "UP" if md > 0 else "DOWN"
        print("    %-10s: biggest=%s in %s (%s %.3f)" % (role, mo, mm, dr, abs(md)))
    print()
    print("-" * 70)
    print("  INTERPRETATION")
    print("-" * 70)
    print()
    print("  A MATRIX:")
    d0 = np.asarray(Al[0]) - np.asarray(Ad[0])
    if np.abs(d0).max() > 0.05:
        v = d0[2, 1]
        if abs(v) > 0.02:
            print("    - Resource AT in MINE: %s (d=%+.3f)" % (
                "more" if v > 0 else "less", v))
    d1 = np.asarray(Al[1]) - np.asarray(Ad[1])
    if np.abs(d1).max() > 0.05:
        v = d1[3, 5]
        if abs(v) > 0.02:
            print("    - Junction in CAPTURE: %s (d=%+.3f)" % (
                "more" if v > 0 else "less", v))
    d2 = np.asarray(Al[2]) - np.asarray(Ad[2])
    r2 = np.sqrt(np.mean(d2**2))
    print("    - Inventory: %s (RMSE=%.4f)" % (
        "drifted" if r2 > 0.01 else "near-deterministic", r2))
    d4 = np.asarray(Al[4]) - np.asarray(Ad[4])
    if np.abs(d4).max() > 0.05:
        print("    - Social obs changed (max|d|=%.3f)" % np.abs(d4).max())
    print()
    print("  B MATRIX:")
    bp = np.asarray(Bl[0]) - np.asarray(Bd[0])
    v = bp[1, 0, 0, 0]
    if abs(v) > 0.02:
        print("    - MINE_CYCLE EXPLORE/EMPTY->MINE: %s (d=%+.3f)" % (
            "more" if v > 0 else "less", v))
    v = bp[3, 0, 0, 1]
    if abs(v) > 0.02:
        print("    - CRAFT_CYCLE EXPLORE/EMPTY->CRAFT: %s (d=%+.3f)" % (
            "more" if v > 0 else "less", v))
    dr2 = np.asarray(Bl[3]) - np.asarray(Bd[3])
    if np.abs(dr2).max() > 0.01:
        print("    - WARNING: Role B changed from identity (max|d|=%.3f)" % (
            np.abs(dr2).max()))
    else:
        print("    - Role B: identity (correct)")
    print()
    print("  C VECTOR:")
    for role in ("miner", "aligner", "scout"):
        shifts = []
        for mi in range(6):
            cd = np.asarray(dc[role][mi])
            cl = np.asarray(rd["C_%s_%d" % (role, mi)])
            d = cl - cd
            for o in range(len(d)):
                if abs(d[o]) > 0.3:
                    shifts.append("%s(%s) %+.2f" % (
                        OL[mi][o], OBS_MODALITY_NAMES[mi], d[o]))
        if shifts:
            print("    %s: %s" % (role, "; ".join(shifts[:5])))
        else:
            print("    %s: minimal (<0.3)" % role)


def main():
    p = argparse.ArgumentParser(description="Analyze learned POMDP params")
    p.add_argument("--params",
        default=str(Path(__file__).resolve().parent.parent / "data" / "learned_joint_deep.npz"))
    p.add_argument("--roles", default=None)
    p.add_argument("--top-k", type=int, default=10)
    a = p.parse_args()
    if a.roles is None:
        a.roles = a.params.replace(".npz", "_roles.npz")
    print("=" * 80)
    print("  LEARNED GENERATIVE MODEL ANALYSIS")
    print("  Params: %s" % a.params)
    print("  Roles:  %s" % a.roles)
    print("=" * 80)
    lrn = np.load(a.params, allow_pickle=True)
    Al = [lrn["A_%d" % i] for i in range(6)]
    Bl = [lrn["B_%d" % i] for i in range(4)]
    meta = str(lrn["metadata"]) if "metadata" in lrn else ""
    rd = np.load(a.roles)
    Ad = build_default_A()
    Bd = build_option_B()
    for m in range(6):
        assert Ad[m].shape == Al[m].shape, "A[%d] mismatch" % m
    for f in range(4):
        assert Bd[f].shape == Bl[f].shape, "B[%d] mismatch" % f
    print()
    print("Shapes verified.")
    print()
    analyze_A(Ad, Al, tk=a.top_k)
    analyze_B(Bd, Bl, tk=a.top_k)
    B_detail(Bd, Bl)
    analyze_C(rd)
    kl_analysis(Ad, Al, Bd, Bl)
    entropy_analysis(Ad, Al, Bd, Bl)
    do_summary(Ad, Al, Bd, Bl, rd, meta)
    print()
    print("Analysis complete.")


if __name__ == "__main__":
    main()
