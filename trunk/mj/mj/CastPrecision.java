// CastPrecision.java, created Sun Feb  8 16:38:30 PST 2004 by gback
// Copyright (C) 2003 Godmar Back <gback@stanford.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
import Compil3r.Analysis.IPA.*;

import org.sf.javabdd.BDD;
import org.sf.javabdd.BDDDomain;
import org.sf.javabdd.BDDFactory;
import org.sf.javabdd.BDDPairing;
import org.sf.javabdd.TypedBDDFactory.TypedBDD;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Comparator;
import Util.Collections.Triple;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.Node;
import Compil3r.Analysis.FlowInsensitive.MethodSummary.CheckCastNode;

import Bootstrap.PrimordialClassLoader;

public class CastPrecision {
    final PAProxy r;
    final PAResults res;
    
    public CastPrecision(PAProxy r, PAResults res) {
        this.r = r;
        this.res = res;
    }

    /**
     * Compute imprecision as manifested by failed casts.
     *
     * In a perfect world, code isn't buggy and every downcast succeeds.
     * In this world, every type-incompatible object that arrives in the
     * points-to set of the input to a downcast is due to imprecision in
     * the points-to analysis.
     *
     * This method looks at each cast and the predecessors nodes leading to it.
     * It computes the vcontext-insensitive points-to sets of both.
     * It counts and reports if a cast always or never succeeds.
     * Casts that always succeed are called "perfect" and they indicate that
     * the results of the analysis and the assumptions of the programmer as
     * to what type of object reach a cast match perfectly.
     *
     * It also reports if a cast's predecessors appear to have empty points-to sets
     * (This would be bad and could indicate an analysis bug.)
     *
     * Those casts that aren't perfect are sorted by the z-ranking of the
     * set sizes of |compatible heap objects|:|all objects reaching predecessors|
     */
    BDD alwaysfail;
    BDD perfectcasts;
    BDD emptycasts;
    BDD casts;

    public void computeCastPrecision(boolean listimperfect) {
        // first step: find all casts
        casts = r.bdd.zero();          // V1
        for (int i = 0; i < r.Vmap.size(); i++) {
            Node n = (Node)r.Vmap.get(i);
            if (n instanceof CheckCastNode) {
                casts.orWith(r.V1.ithVar(i));
            }
        }
        // second step: find all predecessors of casts
        BDD cast_to_pred = r.A.relprod(casts, r.V1cV2cset);     // V1xV1cxV2xV2c & V1 -> V1xV2 
        // compute the points-to sets of predecessors and casts
        BDDPairing V2toV1 = r.bdd.makePair(r.V2, r.V1);
        BDD V1cset = r.V1c.set();
        BDD preds_pt = r.vP.relprod(cast_to_pred.exist(r.V1.set()).replaceWith(V2toV1), V1cset);   // V1xH1xH1c
        BDD casts_pt = r.vP.relprod(cast_to_pred.exist(r.V2.set()), V1cset);                       // V1xH1xH1c
        ArrayList reslist = new ArrayList();

        alwaysfail = (TypedBDD)r.bdd.zero();
        perfectcasts = (TypedBDD)r.bdd.zero();
        emptycasts = (TypedBDD)r.bdd.zero();

        // now look at each cast individually
        for (int i = 0; i < r.Vmap.size(); i++) {
            Node n = (Node)r.Vmap.get(i);
            if (!(n instanceof CheckCastNode))
                continue;
            TypedBDD cast = (TypedBDD)r.V1.ithVar(i);   // V1
            TypedBDD castpt = (TypedBDD)casts_pt.restrict(cast);       // H1xH1c
            BDD preds = cast_to_pred.restrict(cast);    // V2
            preds.replaceWith(V2toV1);                  // V2 -> V1
            TypedBDD predpt = (TypedBDD)preds_pt.relprod(preds, r.V1.set()); // V1xH1xH1c x V1 -> H1xH1c
            if (castpt.equals(predpt)) {
                perfectcasts.orWith(cast);
            } else {
                double compatible = castpt.satCount(r.H1set);
                double all = predpt.satCount(r.H1set);
                if (all == 0.0) {
                    emptycasts.orWith(cast);
                } else {
                    if (compatible == 0.0)
                        alwaysfail.orWith(cast);
                    reslist.add(new Triple(new Double(compatible), new Double(all), "V1("+i+")"+res.getPAResults().longForm(n)));
                }
            }
            preds.free(); predpt.free(); castpt.free();
        }

        if (listimperfect) {
            Object[] rc = reslist.toArray();
            Arrays.sort(rc, new Comparator() {
                public int compare(Object o1, Object o2) {
                    double z1 = zHelper((Triple)o1);
                    double z2 = zHelper((Triple)o2);
                    final int direction = -1;
                    if (Double.isNaN(z1)) return direction;
                    if (Double.isNaN(z2)) return -direction;
                    return z1 < z2 ? direction : z1 > z2 ? -direction : 0;
                }
            });
            for (int i = 0; i < rc.length; i++) {
                System.out.println("#"+i+" z="+zHelper((Triple)rc[i])+" "+rc[i]);
            }
        }
        double castcount = casts.isZero() ? 0.0 : casts.satCount(r.V1.set()); 
        double perfect = perfectcasts.isZero() ? 0.0 : perfectcasts.satCount(r.V1.set()); 
        double notreached = emptycasts.isZero() ? 0.0 : emptycasts.satCount(r.V1.set()); 
        double alwaysfails = alwaysfail.isZero() ? 0.0 : alwaysfail.satCount(r.V1.set()); 
        System.out.println("#casts=" + castcount);
        System.out.println("#perfectcasts (casts that provably always succeed)=" + perfect);
        System.out.println("#perfect-cast ratio=" + (perfect/castcount));
        System.out.println("#emptycasts (casts with empty predecessor points-to set)=" + notreached);
        System.out.println("#alwaysfail (casts with empty compatible points-to sets)=" + alwaysfails);
    }

    private static double zHelper(Triple t) {
        double compatible = ((Double)t.left).doubleValue();
        double all = ((Double)t.middle).doubleValue();
        return PAResults.zcompute(compatible, all-compatible);
    }
}
